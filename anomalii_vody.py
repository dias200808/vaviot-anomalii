# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Аномалии потребления воды — тестовое задание «Вавиот Азия»
#
# **Автор:** Диас Ахметбек · Telegram @akhmetbekdi
#
# **Данные:** 3 000 приборов учёта, 693 443 суточных показания (01.10.2025 – 01.10.2026), 34 009 событий.
#
# **Как устроен ноутбук** — каждый шаг можно перезапустить сверху вниз (`Run All`), все пороги вынесены в один блок `ПОРОГИ`:
#
# 0. Загрузка и проверка данных
# 1. Очистка: шум float, выбросы показаний
# 2. Суточное потребление
# 3. Пункты 1–8 задания
# 4. Итог: кого проверять в первую очередь
#
# Данные (`pokazaniya.csv`, `pribory.csv`, `sobytiya.csv`) кладутся в папку `data/` рядом с ноутбуком.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
DATA = Path('data')

# ---- ПОРОГИ (все в одном месте) ----
ROUND_M3        = 3      # точность показаний: 0.001 м³ = 1 литр
OUTLIER_MIN     = 2      # выброс: показание отличается от медианы 9 соседних больше чем на max(2 м³,
OUTLIER_X       = 20     #         20 обычных суток этого прибора)
BIT_STEP        = 327.68 # 2^15 * 0.01 м³ — шаг «битовой» ошибки (см. находки)
QMAX_DAY        = 75     # физический предел DN15: Q4 = 3.125 м³/ч * 24 ч = 75 м³/сут
SPIKE_ABS       = 3.0    # всплеск: > 3 м³ за сутки (≈ в 25 раз больше среднего дня по базе)
SPIKE_REL       = 10     # всплеск: > 10x обычного дня прибора ...
SPIKE_REL_MIN   = 0.5    # ... и при этом не меньше 0.5 м³ (чтобы 0.01 -> 0.1 не считалось)
LEAK_DAYS       = 90     # утечка: >= 90 суток подряд без единого «нулевого» дня
LEAK_FLOOR      = 0.05   # ... и даже «тихие» дни (5-й перцентиль) >= 50 л/сут
ZERO_DAYS       = 30     # ноль: >= 30 суток подряд расход 0, а пакеты идут
STEP_RATIO      = 3      # ступенька: медиана после / до (или наоборот) >= 3
STEP_MIN_LEVEL  = 0.1    # ... верхний уровень >= 0.1 м³/сут
STEP_MIN_DAYS   = 30     # ... и на каждой стороне >= 30 дней данных

# Палитра графиков (проверенная на дальтонизм)
C_BLUE, C_ORANGE, C_AQUA, C_RED, C_GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#e34948', '#8a8984'
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'axes.edgecolor': '#8a8984', 'axes.axisbelow': True, 'axes.titlesize': 11, 'axes.titleweight': 'bold'})

# %% [markdown]
# ## 0. Загрузка

# %%
P = pd.read_csv(DATA/'pokazaniya.csv', sep=';', dtype={'anon_id': str}, parse_dates=['data'])
D = pd.read_csv(DATA/'pribory.csv',    sep=';', dtype={'anon_id': str})
S = pd.read_csv(DATA/'sobytiya.csv',   sep=';', dtype={'anon_id': str}, parse_dates=['data'])
P = P.sort_values(['anon_id', 'data']).reset_index(drop=True)

print(f'показаний: {len(P):,}  приборов: {P.anon_id.nunique():,}  период: {P.data.min():%d.%m.%Y} – {P.data.max():%d.%m.%Y}')
print('дубликатов (прибор+дата):', P.duplicated(['anon_id', 'data']).sum(), '| пропусков в показаниях:', P.pokazanie.isna().sum())
print('пакетов за сутки: min', P.paketov_za_sutki.min(), '— т.е. каждая строка = прибор был на связи')
print(S.sobytie.value_counts().to_string())

# %% [markdown]
# ## 1. Очистка
#
# **1a. Шум float.** Показания вида `47.585833` после вычитания дают «отрицательный» расход `-3e-13` — это ошибка округления, а не реальный откат. Округляем до 0.001 м³ (1 литр, разрешение счётчика). Это убирает **311** ложных «отрицательных» случаев.
#
# **1b. Выбросы показаний.** Иногда приходит одно-два «мусорных» показания (например `314.83 → 3149.60 → 315.12` или `3.81 → 38.2 → 3.84` — сдвиг запятой), после которых счётчик возвращается на прежний уровень. Накопительный счётчик не может «сходить вверх и вернуться», поэтому показание считаем выбросом, если оно отличается от медианы 9 соседних показаний этого же прибора больше чем на max(2 м³, 20 обычных суток прибора). Настоящий рост (даже большой) не попадает под фильтр: после него показания остаются на новом уровне, и медиана «переезжает» вместе с ними. Такие точки не участвуют в расчёте потребления, но учитываются в пунктах 1 и 3 как ошибки данных.

# %%
P['pokazanie'] = P['pokazanie'].round(ROUND_M3)
med9 = P.groupby('anon_id')['pokazanie'].transform(lambda x: x.rolling(9, center=True, min_periods=3).median())
# «обычные сутки» прибора — медиана положительных приращений (устойчива к сбоям)
_d = P.groupby('anon_id')['pokazanie'].diff()
typ = _d.where(_d > 0).groupby(P.anon_id).transform('median').fillna(0)
thr = np.maximum(OUTLIER_MIN, OUTLIER_X * typ)
# и главное — показание ломает монотонность: после него есть показание ниже, или до него было выше
grp = P.groupby('anon_id')['pokazanie']
fut_min  = grp.transform(lambda x: x[::-1].rolling(4, min_periods=1).min()[::-1].shift(-1))
past_max = grp.transform(lambda x: x.rolling(4, min_periods=1).max().shift(1))
nonmono = (P['pokazanie'] > fut_min + 0.001) | (P['pokazanie'] < past_max - 0.001)
P['vybros'] = ((P['pokazanie'] - med9).abs() > thr) & nonmono

# «сырые» разности (для пунктов 1 и 3 — там нужны именно ошибки)
P['gap']  = P.groupby('anon_id').data.diff().dt.days
P['diff'] = P.groupby('anon_id').pokazanie.diff().round(ROUND_M3)
P['prev_vybros'] = P.groupby('anon_id').vybros.shift(1).fillna(False).astype(bool)
print('выбросов показаний:', P.vybros.sum(), 'у приборов:', P.loc[P.vybros, 'anon_id'].nunique())

# %% [markdown]
# ## 2. Суточное потребление
#
# `расход = показание(день) − показание(предыдущий день)`.
#
# * Если между показаниями пропуск (`gap` > 1 дня — прибор не вышел на связь), разность делится на число дней → средний суточный расход `rate`. Для поиска «нулевых» суток и непрерывных серий берутся **только соседние дни** (`gap == 1`), иначе нельзя сказать, что было в пропущенный день.
# * **Валидный расход:** `0 ≤ rate ≤ 75 м³/сут` (выше — физически невозможно для DN15, см. п.3).

# %%
C = P[~P.vybros].copy()
C['gap']  = C.groupby('anon_id').data.diff().dt.days
C['d']    = C.groupby('anon_id').pokazanie.diff().round(ROUND_M3)
C['rate'] = C['d'] / C['gap']
C = C.dropna(subset=['d'])
C['valid'] = (C['d'] >= 0) & (C['rate'] <= QMAX_DAY)
V  = C[C.valid]                 # все валидные интервалы
V1 = V[V.gap == 1].copy()       # валидные соседние сутки

print(f'интервалов: {len(C):,}, валидных: {len(V):,} ({len(V)/len(C):.1%}), из них соседних суток: {len(V1):,}')
print(f'доля строк с пропуском перед ними: {(P.gap > 1).mean():.1%}')
print(f'средний расход: {V1.d.mean():.3f} м³/сут, медиана {V1.d.median():.3f}, 99-й перцентиль {V1.d.quantile(.99):.2f}')

# приборы с ошибками данных (выбросы или > 75 м³/сут) — их всплески не считаем «водой»
_raw_rate = P['diff'] / P['gap']
err_ids = set(P.loc[P.vybros, 'anon_id']) | set(P.loc[_raw_rate > QMAX_DAY, 'anon_id'])
print('приборов с ошибками данных:', len(err_ids))

# %%
# вспомогательная функция: серии подряд идущих суток, где выполняется условие
def longest_runs(df, cond):
    """Для каждого прибора — самая длинная серия соседних суток (gap==1), где cond == True.
    Возвращает: длину, дату начала, дату конца."""
    x = df[['anon_id', 'data', 'gap', 'd']].copy()
    x['ok'] = cond.values & (x.gap == 1).values
    x['grp'] = (~x.ok).groupby(x.anon_id).cumsum()          # номер серии внутри прибора
    r = (x[x.ok].groupby(['anon_id', 'grp'])
           .agg(days=('data', 'size'), start=('data', 'min'), end=('data', 'max'),
                p5=('d', lambda s: s.quantile(.05)), dmin=('d', 'min'), dmed=('d', 'median')))
    return r.sort_values('days').groupby(level=0).tail(1).droplevel(1)

dev = D.set_index('anon_id')
# «обычный день» прибора = медиана ненулевых валидных суток (если таких >= 30)
pos = V1[V1.d > 0].groupby('anon_id').d.agg(['median', 'size'])
usual = pos.loc[pos['size'] >= 30, 'median']

# %% [markdown]
# ## Пункт 1. Отрицательный расход
#
# **Критерий:** показание меньше предыдущего хотя бы на 1 литр (после округления, т.е. без float-шума).
#
# Разбираем каждый случай по причине:
# * **битовая ошибка** — скачок ровно на ±327.68·k м³ (подробно в находке 1);
# * **возврат после выброса** — предыдущее показание было «мусорным», счётчик вернулся на свой уровень;
# * **сброс/замена** — показание упало почти до 0 (замена прибора или обнуление);
# * **малый откат** до 1 м³ — обратный ток воды / гидроудар / дребезг датчика;
# * **прочее** — крупные необъяснённые падения.

# %%
neg = P[P['diff'] < 0].copy()
k = neg['diff'] / BIT_STEP
neg['prichina'] = np.select(
    [(neg['diff'].abs() > 300) & ((k - k.round()).abs() * BIT_STEP < 1.5),
     neg.prev_vybros,
     (neg.pokazanie < 1) & (neg.pokazanie - neg['diff'] > 5),
     neg['diff'] > -1],
    ['битовая ошибка ±327.68·k', 'возврат после выброса', 'сброс/замена (упал до ~0)', 'малый откат < 1 м³'],
    default='прочее крупное падение')

n1 = neg.groupby('prichina').agg(sluchaev=('diff', 'size'), priborov=('anon_id', 'nunique'),
                                 primery=('anon_id', lambda s: ', '.join(s.value_counts().index[:4])))
print(f'Отрицательный расход: {len(neg)} случаев у {neg.anon_id.nunique()} приборов')
n1.sort_values('sluchaev', ascending=False)

# %%
# связь со «сбросом» из журнала событий
SB = S[S.sobytie == 'сброс'][['anon_id', 'data']].drop_duplicates()
m = neg.merge(SB, on='anon_id', suffixes=('', '_sb'))
m = m[(m.data - m.data_sb).dt.days.abs() <= 3]
print('отрицательных случаев, рядом с которыми (±3 дня) есть событие «сброс»:', m[['anon_id', 'data']].drop_duplicates().shape[0])
neg[neg.prichina == 'малый откат < 1 м³'].sort_values('diff').head(5)[['anon_id', 'data', 'pokazanie', 'diff']]

# %% [markdown]
# ## Пункт 2. Всплески расхода за сутки
#
# Считаем только по валидным соседним суткам (без ошибок данных).
# * **Абсолютный порог:** > 3 м³ за сутки. Средний день по базе ≈ 0.13 м³, 99-й перцентиль ≈ 1.2 м³ — 3 м³ это ~25 «обычных дней».
# * **Относительный порог:** > 10× обычного дня этого прибора (медиана его ненулевых суток) **и** ≥ 0.5 м³ — чтобы скачок с 10 до 100 литров не считался всплеском.
* Приборы с ошибками данных (п.3, находка 1) исключены — у них «всплеск» это сбой, а не вода.

# %%
V1['usual'] = V1.anon_id.map(usual)
V1['x_usual'] = V1.d / V1.usual
V1c = V1[~V1.anon_id.isin(err_ids)]
abs_sp = V1c[V1c.d > SPIKE_ABS]
rel_sp = V1c[(V1c.x_usual > SPIKE_REL) & (V1c.d >= SPIKE_REL_MIN)]
both   = V1c[(V1c.d > SPIKE_ABS) & (V1c.x_usual > SPIKE_REL)]
print(f'абсолютные (> {SPIKE_ABS} м³/сут): {len(abs_sp)} суток у {abs_sp.anon_id.nunique()} приборов')
print(f'относительные (> {SPIKE_REL}x обычного и >= {SPIKE_REL_MIN} м³): {len(rel_sp)} суток у {rel_sp.anon_id.nunique()} приборов')
print(f'оба условия сразу: {len(both)} суток у {both.anon_id.nunique()} приборов')
big = V1c.groupby('anon_id').d.median()
print('NB: у', (big > 3).sum(), 'приборов (', ', '.join(big[big > 3].index), ') расход > 3 м³ почти каждый день — это не всплески, а крупный потребитель (общедомовой/коммерческий узел?)')
both.sort_values('x_usual', ascending=False).head(8)[['anon_id', 'data', 'd', 'usual', 'x_usual']].round(3)

# %% [markdown]
# ## Пункт 3. Нереальные значения
#
# **Физический предел.** У DN15 перегрузочный расход Q4 = 3.125 м³/ч. Даже если открыть всё на максимум и не закрывать сутки — это **75 м³/сут**. Всё, что выше, — не вода, а ошибка данных.
#
# Дополнительно: показание счётчика > 100 000 м³ (квартира за всю жизнь столько не потребит) и «подозрительные» 10–75 м³/сут.

# %%
raw = P.dropna(subset=['diff']).copy()
raw['rate'] = raw['diff'] / raw['gap']
unreal = raw[raw.rate > QMAX_DAY]
k = unreal['diff'] / BIT_STEP
unreal = unreal.assign(tip_oshibki=np.select(
    [((k - k.round()).abs() * BIT_STEP < 1.5), unreal.vybros | unreal.prev_vybros],
    ['битовая ошибка +327.68·k', 'выброс показания'], 'прочее'))
print(f'> {QMAX_DAY} м³/сут: {len(unreal)} интервалов у {unreal.anon_id.nunique()} приборов')
print(unreal.groupby('tip_oshibki').anon_id.agg(['size', 'nunique']))
huge = P[P.pokazanie > 1e5]
print('\nпоказания > 100 000 м³:', huge[['anon_id', 'data', 'pokazanie']].to_string(index=False))
sus = V1[(V1.d > 10)]
print(f'\nвалидные, но сомнительные 10–75 м³/сут: {len(sus)} суток у {sus.anon_id.nunique()} приборов (вероятно общедомовые/коммерческие или ошибка)')
unreal.sort_values('rate', ascending=False).drop_duplicates('anon_id').head(8)[['anon_id', 'data', 'pokazanie', 'diff', 'tip_oshibki']]

# %% [markdown]
# ## Пункт 4. Утечка — прибор «никогда не отдыхает»
#
# В нормальной квартире бывают дни без воды (уехали, на работе, все спят у родных). Утечка (течёт бачок, подтекает кран) даёт **расход каждый день без перерыва**, причём даже в самые «тихие» дни он не падает до нуля.
#
# * **Кандидат:** ≥ 90 соседних суток подряд с расходом > 0.
# * **Сильный признак:** то же + «пол» потребления (5-й перцентиль суточного расхода в этой серии) ≥ 50 л/сут — то есть вода идёт постоянно.
#
# Ограничение: по суточным данным утечку нельзя отличить от большой семьи на 100%. Для точности нужен ночной (часовой) расход — его можно получать с приборов, которые шлют 24+ пакета в сутки.

# %%
leak = longest_runs(C, (C.d > 0) & C.valid)
leak_cand   = leak[leak.days >= LEAK_DAYS]
leak_strong = leak_cand[leak_cand.p5 >= LEAK_FLOOR]
print(f'кандидаты (>= {LEAK_DAYS} суток без нуля): {len(leak_cand)}')
print(f'сильный признак (+ «пол» >= {LEAK_FLOOR*1000:.0f} л/сут): {len(leak_strong)}')
print(f'жёстко (каждый день >= 100 л): {(leak_cand.dmin >= 0.1).sum()}')
leak_strong.sort_values(['p5'], ascending=False).head(10).round(3)

# %% [markdown]
# ## Пункт 5. Нулевое потребление при живом приборе
#
# **Критерий:** ≥ 30 соседних суток подряд расход ровно 0, при этом прибор каждый день на связи (строка есть только если пришёл хотя бы 1 пакет).
#
# Делим на две группы — они означают разное:
# * **никогда не потреблял** (< 100 л за весь год) → пустая квартира / перекрыт стояк / прибор не врезан;
# * **потреблял, потом встал** → уехали надолго **или счётчик застрял/остановлен** — это кандидаты на проверку.

# %%
zero = longest_runs(C, C.d == 0)
zero = zero[zero.days >= ZERO_DAYS].copy()
total = V.groupby('anon_id').d.sum()
zero['vsego_za_god'] = zero.index.map(total).fillna(0)
zero['gruppa'] = np.where(zero.vsego_za_god < 0.1, 'никогда не потреблял', 'потреблял, потом встал')  # < 100 л за год = практически ноль
pk = C.merge(zero[['start', 'end']], left_on='anon_id', right_index=True)
pk = pk[(pk.data >= pk.start) & (pk.data <= pk.end)]
print(f'>= {ZERO_DAYS} суток нуля при живой связи: {len(zero)} приборов; >= 60 суток: {(zero.days >= 60).sum()}')
print('медиана пакетов в сутки в эти дни:', pk.paketov_za_sutki.median())
print(zero.groupby('gruppa').days.agg(['size', 'median', 'max']))
zero[zero.gruppa == 'потреблял, потом встал'].sort_values('days', ascending=False).head(8)

# %% [markdown]
# ## Пункт 6. Ступенька — резкое устойчивое изменение уровня
#
# Простой вариант из задания — сравнить среднее первой и второй половины периода. Но ступенька может быть в любой день, а среднее портит один всплеск. Поэтому:
# * для каждого прибора перебираем точку разреза (≥ 30 дней данных с каждой стороны);
# * сравниваем **медианы** до и после (медиана устойчива к единичным всплескам);
# * ступенька, если уровни различаются ≥ 3 раз и верхний уровень ≥ 0.1 м³/сут.
#
# Тип ступеньки важен: **0 → N** — заселились; **N → 0** — съехали или прибор встал; **N → kN** — больше жильцов или началась утечка; **N → N/k** — меньше людей или вмешательство.

# %%
def best_step(x):
    vals, dates, n = x.values, x.index, len(x)
    best = (0, None, None, None)
    for i in range(STEP_MIN_DAYS, n - STEP_MIN_DAYS + 1, 2):
        m1, m2 = np.median(vals[:i]), np.median(vals[i:])
        if max(m1, m2) < STEP_MIN_LEVEL:
            continue
        r = (max(m1, m2) + 0.01) / (min(m1, m2) + 0.01)
        if r > best[0]:
            best = (r, dates[i], m1, m2)
    return best

rows = []
for aid, g in V.groupby('anon_id'):
    if len(g) < 2 * STEP_MIN_DAYS:
        continue
    x = g.set_index('data').rate
    r, dt, m1, m2 = best_step(x)
    h = len(x) // 2
    rows.append((aid, r, dt, m1, m2, (x.iloc[h:].mean() + .01) / (x.iloc[:h].mean() + .01)))
ST = pd.DataFrame(rows, columns=['anon_id', 'ratio', 'data_stupeni', 'do', 'posle', 'half_ratio']).set_index('anon_id')
step = ST[ST.ratio >= STEP_RATIO].copy()
step['tip'] = np.select([step['do'] < 0.005, step['posle'] < 0.005, step.posle > step['do']],
                        ['0 → N (заселение)', 'N → 0 (выезд / прибор встал)', 'рост N → kN'], 'падение N → N/k')
print(f'ступенька (медианы, >= {STEP_RATIO}x): {len(step)} приборов')
print(f'проверка простым способом из задания (среднее 2-й половины / 1-й >= 3x или <= 1/3): {((ST.half_ratio >= 3) | (ST.half_ratio <= 1/3)).sum()}')
print(step.tip.value_counts().to_string())
step[step.tip.isin(['рост N → kN', 'падение N → N/k'])].sort_values('ratio', ascending=False).head(8).round(3)

# %% [markdown]
# ## Пункт 7. Магнит и потребление
#
# Магнитом пытаются остановить счётчик. Если это кража — в дни «магнита» расход должен падать. Проверяем тремя способами.

# %%
M = S[S.sobytie == 'магнит']
mag_ids = set(M.anon_id)
mday = M[['anon_id', 'data']].drop_duplicates().assign(mag=1)
W = V1[V1.anon_id.isin(mag_ids)].merge(mday, on=['anon_id', 'data'], how='left').fillna({'mag': 0})

print(f'приборов с «магнитом»: {len(mag_ids)} из 3000 ({len(mag_ids)/30:.0f}%)')
print('\n(а) Тот же прибор: дни с магнитом vs без')
print(W.groupby('mag').d.agg(среднее='mean', медиана='median', доля_нулевых=lambda s: (s == 0).mean()).round(3))

pp = W.groupby(['anon_id', 'mag']).d.mean().unstack().dropna()
print(f'   по каждому прибору: у {(pp[1] < pp[0]).mean():.0%} из {len(pp)} в дни магнита расход ниже; медиана отношения {(pp[1]/(pp[0]+1e-9)).median():.2f}')

first = M.groupby('anon_id').data.min()
ba = []
for aid, f in first.items():
    g = V1[V1.anon_id == aid]
    b = g[(g.data < f) & (g.data >= f - pd.Timedelta(days=30))].d
    a = g[(g.data > f) & (g.data <= f + pd.Timedelta(days=30))].d
    if len(b) >= 10 and len(a) >= 10:
        ba.append((aid, b.mean(), a.mean()))
BA = pd.DataFrame(ba, columns=['anon_id', 'do', 'posle']).set_index('anon_id')
print(f'\n(б) 30 дней до первого магнита vs 30 после ({len(BA)} приборов): медиана {BA["do"].median():.3f} -> {BA.posle.median():.3f}; упали больше чем вдвое: {(BA.posle < .5*BA["do"]).mean():.0%}')

base = len(mag_ids) / 3000
grp = {'ноль >= 30 сут (встал)': zero[zero.gruppa == 'потреблял, потом встал'].index,
       'ступенька вниз': step[step.tip.isin(['падение N → N/k', 'N → 0 (выезд / прибор встал)'])].index,
       'утечка (кандидаты)': leak_cand.index, 'отрицательный расход': neg.anon_id.unique()}
print('\n(в) Доля приборов с магнитом в группах аномалий (по базе %.0f%%):' % (base*100))
for k_, ids in grp.items():
    s_ = len(set(ids) & mag_ids) / max(len(ids), 1)
    print(f'   {k_:28s} {s_:5.0%}  (в {s_/base:.1f} раза к базе)')

# %%
# адресно: приборы, у которых в дни магнита расход падает > 5 раз от их нормы
g = W.groupby(['anon_id', 'mag']).d.agg(['mean', 'size']).unstack()
g.columns = ['norma', 'pri_magnite', 'dney_norma', 'dney_magnit']
mag_drop = g[(g.dney_magnit >= 5) & (g.norma >= 0.1) & (g.pri_magnite <= 0.2 * g.norma)]
print('Приборы, где при магните расход падает > 5 раз:')
print(mag_drop.round(3))
print('\nДинамика событий «магнит» по месяцам:')
print(M.groupby(M.data.dt.to_period('M')).size().to_string())
print('\nПриборов с магнитом по типу:')
print(dev.assign(mag=dev.index.isin(mag_ids)).groupby('tip').mag.agg(доля='mean', всего='size').round(2))
print('\nДоля приборов с магнитом по базовым станциям:')
print(dev.assign(mag=dev.index.isin(mag_ids)).groupby('bs').mag.mean().round(2).to_string())

# %% [markdown]
# ## Пункт 8. Собственные находки

# %% [markdown]
# ### Находка 1. Битовая ошибка 327.68 м³ — только у старого типа AQUA2
# Скачки показаний ровно на **+327.68, −327.68, −655.36, −983.04 м³**. 327.68 = 2¹⁵ × 0.01 м³ — счётчик хранит показание в единицах 10 л, и в передаче «переворачивается» 15-й бит. Это не вода, а ошибка прошивки/канала.

# %%
k = raw['diff'] / BIT_STEP
bit = raw[(raw['diff'].abs() > 300) & ((k - k.round()).abs() * BIT_STEP < 1.5)]
bit_ids = bit.anon_id.unique()
print(f'событий: {len(bit)}, приборов: {len(bit_ids)}')
print('распределение по k:', (bit['diff'] / BIT_STEP).round().astype(int).value_counts().sort_index().to_dict())
print('тип приборов с ошибкой:', dev.loc[bit_ids].tip.value_counts().to_dict(), '| всего AQUA2:', (dev.tip == 'AQUA2').sum())
print('базовые станции:', dev.loc[bit_ids].bs.value_counts().to_dict())
pr = P.groupby('anon_id').pokazanie.shift(1)
x10 = P[(pr > 1) & (P.pokazanie / pr).between(9.5, 10.5)]
print(f'сдвиг запятой (показание x10 на сутки): {len(x10)} приборов, тип: {dev.loc[x10.anon_id.unique()].tip.value_counts().to_dict()}')

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={'width_ratios': [1.1, 1]})
# слева: распределение размеров скачков — пики ровно на кратных 327.68
jumps = raw.loc[raw['diff'].abs() > 50, 'diff'].abs().clip(upper=1400)
axes[0].hist(jumps, bins=np.arange(50, 1401, 10), color=C_BLUE)
for kk in (1, 2, 3, 4):
    axes[0].axvline(kk * BIT_STEP, color=C_GRAY, lw=0.8, ls='--')
    axes[0].text(kk * BIT_STEP, axes[0].get_ylim()[1] * 0.92, f'{kk}×327.68', fontsize=8, ha='center', color='#52514e',
                 bbox=dict(facecolor='white', edgecolor='none', pad=1))
axes[0].set_title('Размер скачков показаний > 50 м³', loc='left'); axes[0].set_xlabel('|скачок|, м³'); axes[0].set_ylabel('случаев')
# справа: пример прибора
g = P[(P.anon_id == '8190741') & (P.data.between('2025-12-01', '2026-03-10'))]
axes[1].plot(g.data, g.pokazanie, color=C_BLUE, lw=1.6, drawstyle='steps-post')
axes[1].set_title('Прибор 8190741: уровень прыгает на ±327.68', loc='left'); axes[1].set_ylabel('показание, м³')
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y')); axes[1].tick_params(axis='x', labelsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Находка 2. Взрыв событий «магнит» в августе–сентябре 2026 и «слепой» тип AQUA2
# * До февраля 2026 — около 100 событий в месяц, в сентябре — **19 000**. Расход при этом не упал (п.7) — это похоже на изменение прошивки/чувствительности датчика или логирования, а не на волну краж.
# * У типа **AQUA2 нет ни одного события «магнит»** (0 из 391) — у них нет датчика или он не передаётся. Отсутствие событий у AQUA2 **не значит**, что магнита нет.
* Доля приборов с магнитом сильно зависит от базовой станции: **BS-09 — 56%, BS-07 — 7%**. Воры не живут «по станциям» — это ещё один признак технической природы большинства срабатываний.

# %%
mm = S[S.sobytie == 'магнит'].groupby(S.data.dt.to_period('M')).size()
fig, ax = plt.subplots(figsize=(11, 3))
ax.bar(mm.index.astype(str), mm.values, color=C_ORANGE, width=0.7)
for i, v_ in enumerate(mm.values):
    if v_ > 5000: ax.text(i, v_, f'{v_:,}'.replace(',', ' '), ha='center', va='bottom', fontsize=9)
ax.grid(axis='x', visible=False); ax.set_title('События «магнит» по месяцам', loc='left'); ax.tick_params(axis='x', labelsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Находка 3. Связь и «мёртвые» точки
# * **98 приборов на связи почти весь год (≥ 300 дней), но не потребили и 100 литров** — пустые квартиры, перекрытые стояки или невключённые узлы. Ещё больше «почти мёртвых» среди тех, кто на связи меньше (всего 431 прибор с нулём ≥ 30 дней и < 100 л за всё время).
# * **218 новых приборов на BS-02** появились одним днём — 26.09.2026 (подключили новый дом).
# * Пропуски данных: в среднем у прибора нет **~20%** дней; хуже всего тип `aqua2_nvt_wa_stm` и станции BS-01/05/08 (~69% покрытия) — их стоит проверить по связи.

# %%
span = P.groupby('anon_id').agg(first=('data', 'min'), last=('data', 'max'), n=('data', 'size'))
span['pokrytie'] = span.n / ((span['last'] - span['first']).dt.days + 1)
span = span.join(dev)
dead = total.reindex(dev.index).fillna(0)
print(f'потребили < 100 л за весь год: {(dead < 0.1).sum()} приборов ({(dead < 0.1).mean():.0%}), из них на связи >= 300 дней: {((dead < 0.1) & (span.n.reindex(dev.index) >= 300)).sum()}')
print('новые приборы 26.09.2026:', (span['first'] == '2026-09-26').sum(), span[span['first'] == '2026-09-26'].bs.value_counts().to_dict())
print('\nпокрытие (доля дней с данными) по станциям:')
print(span.groupby('bs').pokrytie.mean().round(2).to_string())
print('\nпо типу:')
print(span.groupby('tip').pokrytie.mean().round(2).to_string())

# %% [markdown]
# ## Примеры аномалий на графиках

# %%
def show(ax, aid, title, color=C_BLUE):
    g = V[V.anon_id == aid]
    ax.plot(g.data, g.rate, color=color, lw=1.2)
    ax.set_title(title, loc='left', fontsize=10); ax.set_ylabel('м³/сут'); ax.tick_params(axis='x', labelsize=8)

ex_leak = leak_strong.sort_values('p5', ascending=False).index[0]
ex_zero = zero[zero.gruppa == 'потреблял, потом встал'].sort_values('days', ascending=False).index[0]
ex_up   = step[step.tip == 'рост N → kN'].sort_values('ratio', ascending=False).index[0]
ex_dn   = step[step.tip == 'падение N → N/k'].sort_values('half_ratio').index[0]  # подтверждается обоими методами
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
show(axes[0, 0], ex_leak, f'Утечка? {ex_leak}: ни одного нулевого дня', C_AQUA)
show(axes[0, 1], ex_zero, f'Встал: {ex_zero}: потреблял, потом ноль', C_RED)
show(axes[1, 0], ex_up,   f'Ступенька вверх: {ex_up}', C_ORANGE)
show(axes[1, 1], ex_dn,   f'Ступенька вниз: {ex_dn}', C_BLUE)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Итог: кого проверять в первую очередь
#
# Баллы (чем выше, тем раньше на выезд):
# * **+3** сильная утечка, которая идёт до сих пор (90+ дней без нуля и «пол» ≥ 50 л/сут) — прямые потери воды; **+1** если утечка была, но закончилась;
# * **+3** при магните расход падает > 5 раз — вероятное вмешательство;
# * **+2** потреблял, потом ≥ 60 суток ноль при живой связи — застрял/остановлен; **+2** если в это время был магнит;
# * **+2** ступенька вниз (N → N/k или N → 0); **+2** если магнит был рядом с датой ступеньки (−30…+60 дней);
# * **+1** ≥ 3 абсолютных всплеска.
#
# Ошибки данных (битовая, x10, > 75 м³/сут) — **отдельный список в техподдержку**, на выезд не нужно.

# %%
score = pd.Series(0, index=dev.index, dtype=float)
why = {i: [] for i in dev.index}
def add(ids, pts, label):
    for i in ids:
        score[i] += pts; why[i].append(label)

stopped = zero[(zero.gruppa == 'потреблял, потом встал') & (zero.days >= 60)]
stopped_mag = [i for i, r in stopped.iterrows()
               if ((M.anon_id == i) & (M.data >= r.start - pd.Timedelta(days=7)) & (M.data <= r.end)).any()]
step_down = step[step.tip.isin(['падение N → N/k', 'N → 0 (выезд / прибор встал)'])]
last_day = P.data.max()
leak_now = leak_strong[leak_strong.end >= last_day - pd.Timedelta(days=14)]
add(leak_now.index, 3, 'утечка идёт сейчас')
add(leak_strong.index.difference(leak_now.index), 1, 'была утечка')
add(mag_drop.index, 3, 'падение при магните')
add(stopped.index, 2, 'встал (60+ дней ноль)')
add(stopped_mag, 2, 'встал + магнит')
add(step_down.index, 2, 'ступенька вниз')
# магнит рядом со ступенькой: от 30 дней до неё до 60 дней после
step_mag = [i for i, r in step_down.iterrows()
            if ((M.anon_id == i) & (M.data >= r.data_stupeni - pd.Timedelta(days=30)) & (M.data <= r.data_stupeni + pd.Timedelta(days=60))).any()]
add(step_mag, 2, 'ступенька вниз + магнит рядом')
sp_cnt = abs_sp.anon_id.value_counts()
add(sp_cnt[sp_cnt >= 3].index, 1, 'всплески')

tech = sorted(set(bit_ids) | set(unreal.anon_id) | set(x10.anon_id) | set(huge.anon_id) | err_ids)
TOP = (pd.DataFrame({'ball': score, 'prichiny': pd.Series({i: ', '.join(v) for i, v in why.items()})})
         .join(dev[['bs', 'tip']]).query('ball > 0').drop(index=tech, errors='ignore')  # ошибки данных — не на выезд
         .sort_values(['ball', 'prichiny'], ascending=False))
print(f'утечка идёт сейчас: {len(leak_now)} | встал >= 60 дней: {len(stopped)} (с магнитом: {len(stopped_mag)}) | ступенька вниз: {len(step_down)} (с магнитом рядом: {len(step_mag)})')
print('приборов с баллом > 0:', len(TOP), '| с баллом >= 4:', (TOP.ball >= 4).sum())
TOP.head(20)

# %%
print(f'В техподдержку (ошибки данных, не вода): {len(tech)} приборов, из них AQUA2: {(dev.loc[tech].tip == "AQUA2").sum()}')
print(', '.join(tech[:20]), '...')

# %%
# сохраняем результаты
Out = Path('results'); Out.mkdir(exist_ok=True)
TOP.to_csv(Out/'na_proverku.csv', sep=';', encoding='utf-8-sig')
dev.loc[tech, ['bs', 'tip', 'model']].to_csv(Out/'v_tehpodderzhku.csv', sep=';', encoding='utf-8-sig')
print('сохранено: results/na_proverku.csv, results/v_tehpodderzhku.csv')

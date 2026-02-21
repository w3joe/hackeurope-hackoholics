"""
LLM Risk Output Extractor
Produces structured text blocks (A, B, C) ready to be passed to an LLM.
No graphs. No EDA. Math logic preserved exactly from notebook.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ripser import ripser as rips


# CONFIG

BASE           = 'https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data/main/data/'
PATHOGEN       = 'Influenza'
COUNTRY        = 'Germany'
FORECAST_WEEKS = 12
WINDOW         = 12        # TDA sliding window size
THRESHOLD      = 2.0       # z-score for anomaly (both dimensions)
STRONG_SOLO    = 3.0       # z-score for single-dimension solo anomaly
ENTROPY_TRAIL  = 8         # how many recent windows to include in B1
RISK_PERCENTILE = 75       # positivity percentile used for risk flagging in C1

FILES = [
    'ILIARIRates.csv',
    'SARIRates.csv',
    'SARITestsDetectionsPositivity.csv',
    'activityFluTypeSubtype.csv',
    'nonSentinelSeverity.csv',
    'nonSentinelTestsDetections.csv',
    'sentinelTestsDetectionsPositivity.csv',
    'sequencingVolumeDetectablePrevalence.csv',
    'variants.csv'
]


# 1. LOAD DATA

print("Loading data from ECDC GitHub...")
dfs = {}
for file in FILES:
    try:
        dfs[file] = pd.read_csv(BASE + file)
        print(f"  {file:<55} {dfs[file].shape}")
    except Exception as e:
        print(f"  {file:<55} ERROR: {e}")


# 2. SENTINEL POSITIVITY 

df = dfs['sentinelTestsDetectionsPositivity.csv'].copy()
df_pos = df[(df['indicator'] == 'positivity') & (df['age'] == 'total')].copy()

coverage = df_pos.groupby('countryname')['yearweek'].nunique().sort_values(ascending=False)
reliable_countries = coverage[coverage >= coverage.median()].index.tolist()

df_reliable = df_pos[df_pos['countryname'].isin(reliable_countries)].copy()

# Multi-pathogen matrix: rows = yearweek, columns = pathogen 
pivot_viruses = df_reliable.pivot_table(
    index='yearweek',
    columns='pathogen',
    values='value',
    aggfunc='median'
)

# 3. TDA ENRICHED MATRIX

sig1 = pivot_viruses.copy()
sig1.columns = [f'sentinel_{c}' for c in sig1.columns]

sig2 = (
    dfs['nonSentinelTestsDetections.csv']
    [
        (dfs['nonSentinelTestsDetections.csv']['age'] == 'total') &
        (dfs['nonSentinelTestsDetections.csv']['indicator'] == 'detections')
    ]
    .pivot_table(index='yearweek', columns='pathogen', values='value', aggfunc='median')
)
tests = (
    dfs['nonSentinelTestsDetections.csv']
    [
        (dfs['nonSentinelTestsDetections.csv']['age'] == 'total') &
        (dfs['nonSentinelTestsDetections.csv']['indicator'] == 'tests')
    ]
    .pivot_table(index='yearweek', columns='pathogen', values='value', aggfunc='median')
)
sig2 = (sig2 / tests * 100).dropna(how='all')
sig2.columns = [f'nonsentinel_{c}' for c in sig2.columns]

tda_matrix_enriched = (
    sig1
    .join(sig2, how='outer')
    .sort_index()
)
thresh = int(tda_matrix_enriched.shape[1] * 0.6)
tda_matrix_enriched = tda_matrix_enriched.dropna(thresh=thresh)
tda_matrix_enriched = tda_matrix_enriched.fillna(tda_matrix_enriched.median())


# 4. PCA 

X_scaled = StandardScaler().fit_transform(tda_matrix_enriched)
weeks     = tda_matrix_enriched.index.tolist()

pca    = PCA(n_components=2)
X_pca  = pca.fit_transform(X_scaled)

# Centroid of all historical points
centroid    = X_pca.mean(axis=0)
pca_dist    = np.sqrt(((X_pca - centroid) ** 2).sum(axis=1))
pca_dist_z  = (pca_dist - pca_dist.mean()) / pca_dist.std()


# 5. TDA SLIDING WINDOW

def persistence_entropy(dgm):
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if lifetimes.sum() == 0:
        return 0
    p = lifetimes / lifetimes.sum()
    return -np.sum(p * np.log(p + 1e-10))

print("Running TDA sliding window")
entropies_h0  = []
entropies_h1  = []
window_labels = []

for i in range(len(X_scaled) - WINDOW + 1):
    window = X_scaled[i : i + WINDOW]
    dgms   = rips(window, maxdim=1)['dgms']
    entropies_h0.append(persistence_entropy(dgms[0]))
    entropies_h1.append(persistence_entropy(dgms[1]))
    window_labels.append(weeks[i + WINDOW - 1])

e0 = np.array(entropies_h0)
e1 = np.array(entropies_h1)
z0 = (e0 - e0.mean()) / e0.std()
z1 = (e1 - e1.mean()) / e1.std()

# Anomaly logic 
both_triggered = (np.abs(z0) > THRESHOLD) & (np.abs(z1) > THRESHOLD)
strong_solo_h0 = np.abs(z0) > STRONG_SOLO
strong_solo_h1 = np.abs(z1) > STRONG_SOLO
anomaly_mask   = both_triggered | strong_solo_h0 | strong_solo_h1
anomaly_idx    = np.where(anomaly_mask)[0]


# 6. SARI / ILI VALIDATION 

sari = (
    dfs['SARIRates.csv']
    [dfs['SARIRates.csv']['age'] == 'total']
    .groupby('yearweek')['value'].median()
    .sort_index()
)

ili = (
    dfs['ILIARIRates.csv']
    [
        (dfs['ILIARIRates.csv']['age'] == 'total') &
        (dfs['ILIARIRates.csv']['indicator'] == 'ILIconsultationrate')
    ]
    .groupby('yearweek')['value'].median()
    .sort_index()
)

sari_trend      = sari.rolling(3, center=True).mean().diff()
tda_anomaly_weeks = set([window_labels[i] for i in anomaly_idx])
rising_weeks    = {w for w in tda_anomaly_weeks if w in sari_trend.index and sari_trend[w] > 0}
falling_weeks   = {w for w in tda_anomaly_weeks if w in sari_trend.index and sari_trend[w] <= 0}


# 7. HOLT-WINTERS — exactly as in HW section of notebook

print("Fitting Holt-Winters...")
series_raw = (
    df_pos[
        (df_pos['pathogen'] == PATHOGEN) &
        (df_pos['countryname'] == COUNTRY)
    ]
    .sort_values('yearweek')
    .set_index('yearweek')['value']
    .dropna()
)

series_raw.index = pd.to_datetime(
    [f"{yw}-1" for yw in series_raw.index],
    format="%G-W%V-%u"
)

# Build a complete weekly grid — required for HW on countries with reporting gaps
full_index          = pd.date_range(start=series_raw.index.min(), end=series_raw.index.max(), freq='W-MON')
series_raw_reindexed = series_raw.reindex(full_index)

# Detect gaps before filling so we can warn the LLM
gap_mask  = series_raw_reindexed.isna()
gap_count = int(gap_mask.sum())
gap_runs  = []
in_gap    = False
gap_start = None

for date, is_gap in zip(series_raw_reindexed.index, gap_mask):
    if is_gap and not in_gap:
        in_gap    = True
        gap_start = date
    elif not is_gap and in_gap:
        in_gap = False
        gap_runs.append((gap_start, date - pd.Timedelta(weeks=1)))
if in_gap:
    gap_runs.append((gap_start, series_raw_reindexed.index[-1]))

series_raw = series_raw_reindexed.interpolate(method='linear')
covid_cutoff = pd.Timestamp('2022-01-01')
clean_series = series_raw[series_raw.index >= covid_cutoff]

if len(clean_series) >= 2 * 52:
    series = clean_series
    TRAIN_START = str(covid_cutoff.date())
    hw_data_warning = None
else:
    series = series_raw
    TRAIN_START = str(series_raw.index.min().date())
    hw_data_warning = (
        f"Not enough data from {covid_cutoff.date()} ({len(clean_series)} weeks). "
        f"Using full available history from {TRAIN_START} ({len(series)} weeks) instead."
    )

results = {}
for seasonal_type in ['add', 'mul']:
    try:
        m = ExponentialSmoothing(
            series,
            trend='add',
            seasonal=seasonal_type,
            seasonal_periods=52,
            initialization_method='estimated'
        ).fit(optimized=True)
        results[seasonal_type] = m
    except Exception as e:
        print(f"[{seasonal_type}] failed: {e}")

if not results:
    hw_unavailable = True
    forecast  = None
    fitted    = None
    residuals = None
    residual_std = None
    residual_z   = None
    hw_anomaly_weeks = []
    hw_risk_weeks    = []
    best_type = None
else:
    hw_unavailable = False
    best_type = min(results, key=lambda k: results[k].aic)
    fit       = results[best_type]
    forecast  = fit.forecast(FORECAST_WEEKS)
    fitted    = fit.fittedvalues
    residuals     = series.values - fitted.values
    residual_std  = residuals.std()
    residual_z    = residuals / residual_std

# residuals     = series.values - fitted.values
# residual_std  = residuals.std()
# residual_z    = residuals / residual_std

risk_threshold = np.percentile(series.values, RISK_PERCENTILE)


# OUTPUT ASSEMBLY 


output_lines = []

def section(title):
    output_lines.append("")
    output_lines.append(title)


def subsection(title):
    output_lines.append("")
    output_lines.append(f"{title}")


# BLOCK A: HISTORICAL ANOMALY RECORD

section("BLOCK A: HISTORICAL ANOMALY RECORD")

# A1. Holt-Winters residual anomalies
subsection("A1. Holt-Winters Residual Anomalies (past weeks where model was surprised)")
if hw_unavailable:
    output_lines.append(f"Holt-Winters model could not be fitted for {COUNTRY}/{PATHOGEN}.")
    output_lines.append(f"  Possible reasons: fewer than 104 weeks of data available after fallback to full history, or series contains zero values which prevent multiplicative fitting.")
    output_lines.append(f"  Available weeks: {len(series)}")
    output_lines.append(f"  No residual anomalies, forecast, or HW-based risk flags can be provided for this combination.")
    output_lines.append(f"  Risk assessment for this pathogen/country relies entirely on TDA (Blocks A2, B1, B2) and cross-validation (A3).")
else:
    output_lines.append(f"Pathogen: {PATHOGEN} | Country: {COUNTRY} | Model: seasonal='{best_type}' | AIC={fit.aic:.1f}")
    if hw_data_warning:
        output_lines.append(hw_data_warning)
    output_lines.append(f"Residual std: {residual_std:.3f} | Risk threshold (p{RISK_PERCENTILE}): {risk_threshold:.2f}%")

    if gap_count == 0:
        output_lines.append("Data quality: no missing weeks — series is complete.")
    else:
        output_lines.append(f"DATA QUALITY WARNING: {gap_count} missing week(s) in {COUNTRY} reporting — filled via linear interpolation.")
        output_lines.append(f"  Treat HW output in these periods with lower confidence:")
        for start, end in gap_runs:
            n_weeks = int((end - start).days / 7) + 1
            output_lines.append(f"    {start.date()} → {end.date()}  ({n_weeks} week(s))")
        if any((end - start).days / 7 >= 4 for start, end in gap_runs):
            output_lines.append("At least one gap ≥4 weeks — model seasonality in this period is unreliable.")
    output_lines.append("")
    output_lines.append(f"{'week':<14} {'observed':>10} {'fitted':>10} {'residual':>10} {'z_score':>9} {'direction'}")

    hw_anomaly_weeks = series.index[np.abs(residual_z) > 2.0]
    for date, obs, fit_val, res, z in zip(
        series.index, series.values, fitted.values, residuals, residual_z
    ):
        if abs(z) > 2.0:
            direction = "SURGE" if res > 0 else "DROP"
            output_lines.append(
                f"{str(date.date()):<14} {obs:>10.2f} {fit_val:>10.2f} {res:>10.2f} {z:>+9.2f}  {direction}"
            )

    output_lines.append(f"\nTotal past HW anomaly weeks: {len(hw_anomaly_weeks)}")

# A2. TDA anomaly windows
subsection("A2. TDA Topological Anomaly Windows (strict: both>2σ OR solo>3σ)")
output_lines.append(f"Window size: {WINDOW} weeks | Total windows computed: {len(window_labels)}")
output_lines.append(f"Anomaly logic: (|z0|>{THRESHOLD} AND |z1|>{THRESHOLD}) OR |z0|>{STRONG_SOLO} OR |z1|>{STRONG_SOLO}")
output_lines.append("")
output_lines.append(f"{'window_end_week':<18} {'H0_z':>8} {'H1_z':>8}  trigger")

for idx in anomaly_idx:
    trigger = []
    if both_triggered[idx]: trigger.append("both>2σ")
    if strong_solo_h0[idx]: trigger.append("H0-solo>3σ")
    if strong_solo_h1[idx]: trigger.append("H1-solo>3σ")
    output_lines.append(
        f"{window_labels[idx]:<18} {z0[idx]:>+8.2f} {z1[idx]:>+8.2f}  {', '.join(trigger)}"
    )

output_lines.append(f"\nTotal TDA anomaly windows: {len(anomaly_idx)}")

# A3. SARI/ILI cross-validation
subsection("A3. SARI/ILI Validation of TDA Anomaly Weeks")
output_lines.append("Direction based on 3-week smoothed SARI trend at the anomaly week.")
output_lines.append(f"SARI median (all weeks):          {sari.median():.2f}")
output_lines.append(f"SARI median at rising anomalies:  {sari[[w for w in rising_weeks if w in sari.index]].median():.2f}")
output_lines.append(f"SARI median at falling anomalies: {sari[[w for w in falling_weeks if w in sari.index]].median():.2f}")
output_lines.append(f"ILI median (all weeks):           {ili.median():.2f}")
output_lines.append(f"ILI median at rising anomalies:   {ili[[w for w in rising_weeks if w in ili.index]].median():.2f}")
output_lines.append(f"ILI median at falling anomalies:  {ili[[w for w in falling_weeks if w in ili.index]].median():.2f}")
output_lines.append("")
output_lines.append(f"{'week':<18} {'direction':<12} {'sari_value':>12} {'ili_value':>12}")

for w in sorted(tda_anomaly_weeks):
    direction = "RISING" if w in rising_weeks else "FALLING"
    sari_val  = f"{sari[w]:.2f}" if w in sari.index else "n/a"
    ili_val   = f"{ili[w]:.2f}"  if w in ili.index  else "n/a"
    output_lines.append(f"{w:<18} {direction:<12} {sari_val:>12} {ili_val:>12}")

# A4. Seasonal pattern (ISO week frequency of anomalies)
subsection("A4. Seasonal Pattern — ISO Week Distribution of TDA Anomalies")
output_lines.append("Frequency of anomaly window end-weeks by ISO week number:")
output_lines.append("(Shows which weeks of the year historically concentrate anomalies)")
output_lines.append("")

iso_weeks = [int(w.split('-W')[1]) if '-W' in str(w) else pd.Timestamp(w).isocalendar()[1]
             for w in tda_anomaly_weeks]
iso_counter = {}
for w in iso_weeks:
    iso_counter[w] = iso_counter.get(w, 0) + 1

for iso_w in sorted(iso_counter):
    output_lines.append(f"  ISO week {iso_w:>2}: ({iso_counter[iso_w]})")


# BLOCK B CURRENT STATE

section("BLOCK B: CURRENT STATE (most recent observations)")

# B1. TDA entropy trajectory (last N windows)
subsection(f"B1. TDA Entropy Trajectory: Last {ENTROPY_TRAIL} Windows")
output_lines.append("Read the direction of travel: is entropy climbing toward the anomaly threshold?")
output_lines.append(f"Threshold: ±{THRESHOLD}σ (both) | ±{STRONG_SOLO}σ (solo)")
output_lines.append("")
output_lines.append(f"{'window_end_week':<18} {'H0_z':>8} {'H1_z':>8}  status")

trail_indices = list(range(max(0, len(window_labels) - ENTROPY_TRAIL), len(window_labels)))
for idx in trail_indices:
    if anomaly_mask[idx]:
        status = "ANOMALY"
    elif abs(z0[idx]) > 1.5 or abs(z1[idx]) > 1.5:
        status = "elevated"
    else:
        status = "normal"
    output_lines.append(
        f"{window_labels[idx]:<18} {z0[idx]:>+8.2f} {z1[idx]:>+8.2f}  {status}"
    )

# Compute drift direction over the trail
recent_z0 = z0[trail_indices]
recent_z1 = z1[trail_indices]
drift_z0 = "RISING" if recent_z0[-1] > recent_z0[0] else "FALLING"
drift_z1 = "RISING" if recent_z1[-1] > recent_z1[0] else "FALLING"
output_lines.append("")
output_lines.append(f"H0 entropy trend over last {ENTROPY_TRAIL} windows: {drift_z0}  "
                    f"({recent_z0[0]:+.2f} → {recent_z0[-1]:+.2f})")
output_lines.append(f"H1 entropy trend over last {ENTROPY_TRAIL} windows: {drift_z1}  "
                    f"({recent_z1[0]:+.2f} → {recent_z1[-1]:+.2f})")

current_anomalous = anomaly_mask[trail_indices[-1]]
output_lines.append(f"Most recent window anomalous: {'YES ' if current_anomalous else 'NO'}")

# B2. PCA current positioning
subsection("B2. PCA Positioning: Last 6 Weeks vs Historical Centroid")
output_lines.append(f"Variance explained by 2 PCs: {sum(pca.explained_variance_ratio_)*100:.1f}%")
output_lines.append(f"Historical mean distance from centroid: {pca_dist.mean():.3f}  std: {pca_dist.std():.3f}")
output_lines.append("")
output_lines.append(f"{'week':<18} {'PC1':>8} {'PC2':>8} {'dist_centroid':>15} {'dist_z':>8}  position")

for i in range(max(0, len(weeks) - 6), len(weeks)):
    w      = weeks[i]
    pc1    = X_pca[i, 0]
    pc2    = X_pca[i, 1]
    dist   = pca_dist[i]
    dz     = pca_dist_z[i]
    pos    = "FAR from normal" if dz > 2.0 else ("elevated" if dz > 1.0 else "within normal range")
    output_lines.append(f"{w:<18} {pc1:>8.3f} {pc2:>8.3f} {dist:>15.3f} {dz:>+8.2f}  {pos}")

# B3. Most recent HW residual
subsection("B3. Holt-Winters: Most Recent Fitted Residual")
if hw_unavailable:
    output_lines.append(f"Holt-Winters model could not be fitted for {COUNTRY}/{PATHOGEN}.")
    output_lines.append(f"  Possible reasons: fewer than 104 weeks of data available after fallback to full history, or series contains zero values which prevent multiplicative fitting.")
    output_lines.append(f"  Available weeks: {len(series)}")
    output_lines.append(f"  No residual anomalies, forecast, or HW-based risk flags can be provided for this combination.")
    output_lines.append(f"  Risk assessment for this pathogen/country relies entirely on TDA (Blocks A2, B1, B2) and cross-validation (A3).")
else:
    last_date    = series.index[-1]
    last_obs     = series.values[-1]
    last_fitted  = fitted.values[-1]
    last_resid   = residuals[-1]
    last_z       = residual_z[-1]
    output_lines.append(f"Most recent observed week : {last_date.date()}")
    output_lines.append(f"Observed positivity       : {last_obs:.2f}%")
    output_lines.append(f"Holt-Winters fitted value : {last_fitted:.2f}%")
    output_lines.append(f"Residual                  : {last_resid:+.2f}%")
    output_lines.append(f"Residual z-score          : {last_z:+.2f}")
    output_lines.append(f"Model currently surprised : {'YES ' if abs(last_z) > 2 else 'NO'}")


# BLOCK C FUTURE OUTLOOK

section("BLOCK C — FUTURE OUTLOOK")

# C1. Holt-Winters forecast table
subsection(f"C1. Holt-Winters Forecast: Next {FORECAST_WEEKS} Weeks")
if hw_unavailable:
    output_lines.append(f"Holt-Winters model could not be fitted for {COUNTRY}/{PATHOGEN}.")
    output_lines.append(f"  Possible reasons: fewer than 104 weeks of data available after fallback to full history, or series contains zero values which prevent multiplicative fitting.")
    output_lines.append(f"  Available weeks: {len(series)}")
    output_lines.append(f"  No residual anomalies, forecast, or HW-based risk flags can be provided for this combination.")
    output_lines.append(f"  Risk assessment for this pathogen/country relies entirely on TDA (Blocks A2, B1, B2) and cross-validation (A3).")
else:
    output_lines.append(f"Pathogen: {PATHOGEN} | Country: {COUNTRY}")
    output_lines.append(f"Risk flag threshold: >{risk_threshold:.2f}% (historical p{RISK_PERCENTILE})")
    output_lines.append("")
    output_lines.append(f"{'week':<14} {'forecast_%':>12} {'risk_flag'}")

    hw_risk_weeks = []
    for date, val in zip(forecast.index, forecast.values):
        flag = "HIGH RISK " if val >= risk_threshold else "normal"
        if val >= risk_threshold:
            hw_risk_weeks.append(str(date.date()))
        output_lines.append(f"{str(date.date()):<14} {val:>12.2f}  {flag}")

# C2. TDA entropy trend projection
subsection("C2. TDA Entropy Trend: Forward Signal")
output_lines.append("TDA does not forecast directly. Read the recent entropy trajectory")
output_lines.append("to assess whether the system is moving toward or away from anomaly.")
output_lines.append("")

# Historical average lead time: weeks between entropy crossing 1.5σ and confirmed anomaly
lead_times = []
for idx in anomaly_idx:
    # Look back to find when entropy first crossed 1.5σ before this anomaly
    for lookback in range(1, WINDOW + 1):
        prev_idx = idx - lookback
        if prev_idx >= 0 and abs(z0[prev_idx]) <= 1.5 and abs(z1[prev_idx]) <= 1.5:
            lead_times.append(lookback)
            break

avg_lead = int(np.mean(lead_times)) if lead_times else 4
output_lines.append(f"Historical avg lead time (from elevation >1.5σ to confirmed anomaly): ~{avg_lead} weeks")
output_lines.append("")

last_z0 = z0[trail_indices[-1]]
last_z1 = z1[trail_indices[-1]]

if current_anomalous:
    output_lines.append("STATUS: System is currently IN an anomalous topological state.")
    output_lines.append("Risk is IMMEDIATE. Monitor HW forecast convergence closely.")
elif abs(last_z0) > 1.5 or abs(last_z1) > 1.5:
    output_lines.append(f"STATUS: Entropy is ELEVATED but below anomaly threshold.")
    output_lines.append(f"H0_z={last_z0:+.2f}  H1_z={last_z1:+.2f}")
    output_lines.append(f"If trend continues, anomaly could materialize in ~{avg_lead} weeks.")
    output_lines.append(f"Projected risk window: approximately week {avg_lead} from {window_labels[-1]}.")
else:
    output_lines.append(f"STATUS: Entropy within normal range.")
    output_lines.append(f"H0_z={last_z0:+.2f}  H1_z={last_z1:+.2f}")
    output_lines.append("No imminent topological risk signal from TDA at this time.")

# C3. Convergence flags
subsection("C3. Convergence Flags — Weeks Where Both Methods Agree")
if hw_unavailable:
    output_lines.append("HW unavailable: see reason above.")
else:
    output_lines.append("Highest-confidence risk: Holt-Winters forecasts HIGH RISK AND TDA signals anomaly/elevation.")
    output_lines.append("")

    # Map HW risk weeks to ISO-week format for comparison
    hw_risk_iso = set()
    for date_str in hw_risk_weeks:
        d = pd.Timestamp(date_str)
        hw_risk_iso.add(f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}")

    # TDA risk context: anomaly OR elevated trajectory pointing forward
    tda_currently_elevated = current_anomalous or abs(last_z0) > 1.5 or abs(last_z1) > 1.5

    convergence_found = False
    output_lines.append(f"{'forecast_week':<16} {'hw_positivity':>14} {'tda_context':<35} {'confidence'}")

    for date, val in zip(forecast.index, forecast.values):
        iso_str = f"{date.isocalendar()[0]}-W{date.isocalendar()[1]:02d}"
        hw_high = val >= risk_threshold

        # TDA context for this week
        # If currently in anomaly or elevated AND within avg_lead window
        weeks_ahead = list(forecast.index).index(date) + 1
        tda_context_flag = tda_currently_elevated and weeks_ahead <= avg_lead + 2

        if hw_high and tda_context_flag:
            confidence   = "(3) HIGH CONVERGENCE"
            convergence_found = True
        elif hw_high:
            confidence   = "(2) HW only"
        elif tda_context_flag:
            confidence   = "(1) TDA signal only"
        else:
            confidence   = "(0) no signal"

        tda_desc = "TDA: elevated/anomaly -> risk window" if tda_context_flag else "TDA: normal"
        output_lines.append(
            f"{str(date.date()):<16} {val:>14.2f}  {tda_desc:<35} {confidence}"
        )

    if not convergence_found:
        output_lines.append("")
        output_lines.append("No full convergence found in forecast window.")
        output_lines.append("This means HW and TDA risk windows do not overlap in the next 12 weeks.")


# PRINT + SAVE

full_output = "\n".join(output_lines)
print(full_output)

output_path = 'llm_risk_output.txt'
with open(output_path, 'w') as f:
    f.write(full_output)

print(f"\n\nOutput saved to: {output_path}")
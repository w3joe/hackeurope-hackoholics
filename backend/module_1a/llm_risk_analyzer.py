"""
LLM Risk Output Extractor for Module 1A.
Produces structured text blocks (A, B, C) for every pathogen×country combination.
No graphs. Math logic preserved from Trend_Analysis/LLM_Output.py.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ripser import ripser as rips

# Config
BASE = 'https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data/main/data/'
FORECAST_WEEKS = 12
WINDOW = 12
THRESHOLD = 2.0
STRONG_SOLO = 3.0
ENTROPY_TRAIL = 8
RISK_PERCENTILE = 75

FILES = [
    'ILIARIRates.csv',
    'SARIRates.csv',
    'SARITestsDetectionsPositivity.csv',
    'activityFluTypeSubtype.csv',
    'nonSentinelSeverity.csv',
    'nonSentinelTestsDetections.csv',
    'sentinelTestsDetectionsPositivity.csv',
    'sequencingVolumeDetectablePrevalence.csv',
    'variants.csv',
]


def _persistence_entropy(dgm):
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if lifetimes.sum() == 0:
        return 0
    p = lifetimes / lifetimes.sum()
    return -np.sum(p * np.log(p + 1e-10))


def load_data():
    """Load ECDC data from GitHub."""
    dfs = {}
    for file in FILES:
        try:
            dfs[file] = pd.read_csv(BASE + file)
        except Exception as e:
            print(f"  {file}: ERROR {e}")
    return dfs


def compute_shared_tda(dfs):
    """Compute TDA, PCA, SARI/ILI once (shared across all pathogen/country)."""
    df = dfs['sentinelTestsDetectionsPositivity.csv'].copy()
    df_pos = df[(df['indicator'] == 'positivity') & (df['age'] == 'total')].copy()

    coverage = df_pos.groupby('countryname')['yearweek'].nunique().sort_values(ascending=False)
    reliable_countries = coverage[coverage >= coverage.median()].index.tolist()
    df_reliable = df_pos[df_pos['countryname'].isin(reliable_countries)].copy()

    pivot_viruses = df_reliable.pivot_table(
        index='yearweek', columns='pathogen', values='value', aggfunc='median'
    )
    sig1 = pivot_viruses.copy()
    sig1.columns = [f'sentinel_{c}' for c in sig1.columns]

    ns = dfs['nonSentinelTestsDetections.csv']
    sig2 = (
        ns[(ns['age'] == 'total') & (ns['indicator'] == 'detections')]
        .pivot_table(index='yearweek', columns='pathogen', values='value', aggfunc='median')
    )
    tests = (
        ns[(ns['age'] == 'total') & (ns['indicator'] == 'tests')]
        .pivot_table(index='yearweek', columns='pathogen', values='value', aggfunc='median')
    )
    sig2 = (sig2 / tests * 100).dropna(how='all')
    sig2.columns = [f'nonsentinel_{c}' for c in sig2.columns]

    tda_matrix_enriched = sig1.join(sig2, how='outer').sort_index()
    thresh = int(tda_matrix_enriched.shape[1] * 0.6)
    tda_matrix_enriched = tda_matrix_enriched.dropna(thresh=thresh)
    tda_matrix_enriched = tda_matrix_enriched.fillna(tda_matrix_enriched.median())

    X_scaled = StandardScaler().fit_transform(tda_matrix_enriched)
    weeks = tda_matrix_enriched.index.tolist()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroid = X_pca.mean(axis=0)
    pca_dist = np.sqrt(((X_pca - centroid) ** 2).sum(axis=1))
    pca_dist_z = (pca_dist - pca_dist.mean()) / pca_dist.std()

    entropies_h0, entropies_h1, window_labels = [], [], []
    for i in range(len(X_scaled) - WINDOW + 1):
        window = X_scaled[i : i + WINDOW]
        dgms = rips(window, maxdim=1)['dgms']
        entropies_h0.append(_persistence_entropy(dgms[0]))
        entropies_h1.append(_persistence_entropy(dgms[1]))
        window_labels.append(weeks[i + WINDOW - 1])

    e0 = np.array(entropies_h0)
    e1 = np.array(entropies_h1)
    z0 = (e0 - e0.mean()) / e0.std()
    z1 = (e1 - e1.mean()) / e1.std()

    both_triggered = (np.abs(z0) > THRESHOLD) & (np.abs(z1) > THRESHOLD)
    strong_solo_h0 = np.abs(z0) > STRONG_SOLO
    strong_solo_h1 = np.abs(z1) > STRONG_SOLO
    anomaly_mask = both_triggered | strong_solo_h0 | strong_solo_h1
    anomaly_idx = np.where(anomaly_mask)[0]

    sari = (
        dfs['SARIRates.csv'][dfs['SARIRates.csv']['age'] == 'total']
        .groupby('yearweek')['value'].median().sort_index()
    )
    ili = (
        dfs['ILIARIRates.csv'][
            (dfs['ILIARIRates.csv']['age'] == 'total') &
            (dfs['ILIARIRates.csv']['indicator'] == 'ILIconsultationrate')
        ]
        .groupby('yearweek')['value'].median().sort_index()
    )
    sari_trend = sari.rolling(3, center=True).mean().diff()
    tda_anomaly_weeks = set(window_labels[i] for i in anomaly_idx)
    rising_weeks = {w for w in tda_anomaly_weeks if w in sari_trend.index and sari_trend[w] > 0}
    falling_weeks = {w for w in tda_anomaly_weeks if w in sari_trend.index and sari_trend[w] <= 0}

    return {
        'df_pos': df_pos,
        'weeks': weeks,
        'X_pca': X_pca,
        'pca': pca,
        'pca_dist': pca_dist,
        'pca_dist_z': pca_dist_z,
        'window_labels': window_labels,
        'z0': z0,
        'z1': z1,
        'anomaly_idx': anomaly_idx,
        'anomaly_mask': anomaly_mask,
        'both_triggered': both_triggered,
        'strong_solo_h0': strong_solo_h0,
        'strong_solo_h1': strong_solo_h1,
        'sari': sari,
        'ili': ili,
        'tda_anomaly_weeks': tda_anomaly_weeks,
        'rising_weeks': rising_weeks,
        'falling_weeks': falling_weeks,
    }


def _fit_holt_winters(df_pos, pathogen, country, end_date=None):
    """Fit Holt-Winters for a single pathogen/country.
    If end_date (datetime) is given, fit only on data up to that date (for historical simulation)."""
    series_raw = (
        df_pos[(df_pos['pathogen'] == pathogen) & (df_pos['countryname'] == country)]
        .sort_values('yearweek')
        .set_index('yearweek')['value']
        .dropna()
    )
    if len(series_raw) < 2:
        return {'unavailable': True, 'series_len': len(series_raw), 'series': None}

    series_raw.index = pd.to_datetime(
        [f"{yw}-1" for yw in series_raw.index],
        format="%G-W%V-%u"
    )
    full_index = pd.date_range(start=series_raw.index.min(), end=series_raw.index.max(), freq='W-MON')
    series_raw_reindexed = series_raw.reindex(full_index)

    gap_mask = series_raw_reindexed.isna()
    gap_count = int(gap_mask.sum())
    gap_runs = []
    in_gap, gap_start = False, None
    for date, is_gap in zip(series_raw_reindexed.index, gap_mask):
        if is_gap and not in_gap:
            in_gap, gap_start = True, date
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
        hw_data_warning = None
    else:
        series = series_raw
        hw_data_warning = (
            f"Not enough data from {covid_cutoff.date()} ({len(clean_series)} weeks). "
            f"Using full available history ({len(series)} weeks) instead."
        )

    full_series = series.copy()
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        series = series[series.index <= end_ts]
        if len(series) < 52:
            return {'unavailable': True, 'series_len': len(series), 'series': series}

    results = {}
    for seasonal_type in ['add', 'mul']:
        try:
            m = ExponentialSmoothing(
                series, trend='add', seasonal=seasonal_type,
                seasonal_periods=52, initialization_method='estimated'
            ).fit(optimized=True)
            results[seasonal_type] = m
        except Exception:
            pass

    if not results:
        return {
            'unavailable': True, 'series': series,
            'gap_count': gap_count, 'gap_runs': gap_runs,
            'hw_data_warning': hw_data_warning,
        }

    best_type = min(results, key=lambda k: results[k].aic)
    fit = results[best_type]
    forecast = fit.forecast(FORECAST_WEEKS)
    fitted = fit.fittedvalues
    residuals = series.values - fitted.values
    residual_std = residuals.std()
    residual_z = residuals / residual_std
    risk_threshold = np.percentile(series.values, RISK_PERCENTILE)
    hw_anomaly_weeks = series.index[np.abs(residual_z) > 2.0]
    hw_risk_weeks = [str(d.date()) for d, v in zip(forecast.index, forecast.values) if v >= risk_threshold]

    out = {
        'unavailable': False,
        'series': series,
        'fit': fit,
        'best_type': best_type,
        'forecast': forecast,
        'fitted': fitted,
        'residuals': residuals,
        'residual_std': residual_std,
        'residual_z': residual_z,
        'risk_threshold': risk_threshold,
        'hw_anomaly_weeks': hw_anomaly_weeks,
        'hw_risk_weeks': hw_risk_weeks,
        'gap_count': gap_count,
        'gap_runs': gap_runs,
        'hw_data_warning': hw_data_warning,
    }
    if end_date is not None:
        out['full_series'] = full_series
    return out


def run_single_pathogen_country(df_pos, shared, pathogen, country):
    """Generate LLM output blocks for one pathogen/country."""
    PATHOGEN, COUNTRY = pathogen, country
    output_lines = []

    def section(t): output_lines.append(""); output_lines.append(t)
    def subsection(t): output_lines.append(""); output_lines.append(t)

    hw = _fit_holt_winters(df_pos, pathogen, country)
    hw_unavailable = hw['unavailable']

    section("BLOCK A: HISTORICAL ANOMALY RECORD")
    subsection("A1. Holt-Winters Residual Anomalies (past weeks where model was surprised)")
    if hw_unavailable:
        output_lines.append(f"Holt-Winters model could not be fitted for {COUNTRY}/{PATHOGEN}.")
        output_lines.append("  Possible reasons: fewer than 104 weeks of data, or series contains zero values.")
        avail = hw.get('series_len')
        if avail is None and hw.get('series') is not None:
            avail = len(hw['series'])
        output_lines.append(f"  Available weeks: {avail if avail is not None else 'n/a'}")
        output_lines.append("  Risk assessment relies on TDA (Blocks A2, B1, B2) and cross-validation (A3).")
    else:
        fit, best_type = hw['fit'], hw['best_type']
        output_lines.append(f"Pathogen: {PATHOGEN} | Country: {COUNTRY} | Model: seasonal='{best_type}' | AIC={fit.aic:.1f}")
        if hw.get('hw_data_warning'):
            output_lines.append(hw['hw_data_warning'])
        output_lines.append(f"Residual std: {hw['residual_std']:.3f} | Risk threshold (p{RISK_PERCENTILE}): {hw['risk_threshold']:.2f}%")
        gap_count = hw['gap_count']
        if gap_count == 0:
            output_lines.append("Data quality: no missing weeks — series is complete.")
        else:
            output_lines.append(f"DATA QUALITY WARNING: {gap_count} missing week(s) — filled via linear interpolation.")
            for start, end in hw['gap_runs']:
                n_weeks = int((end - start).days / 7) + 1
                output_lines.append(f"    {start.date()} → {end.date()}  ({n_weeks} week(s))")
        output_lines.append("")
        output_lines.append(f"{'week':<14} {'observed':>10} {'fitted':>10} {'residual':>10} {'z_score':>9} {'direction'}")
        for date, obs, fit_val, res, z in zip(
            hw['series'].index, hw['series'].values, hw['fitted'].values,
            hw['residuals'], hw['residual_z']
        ):
            if abs(z) > 2.0:
                direction = "SURGE" if res > 0 else "DROP"
                output_lines.append(f"{str(date.date()):<14} {obs:>10.2f} {fit_val:>10.2f} {res:>10.2f} {z:>+9.2f}  {direction}")
        output_lines.append(f"\nTotal past HW anomaly weeks: {len(hw['hw_anomaly_weeks'])}")

    subsection("A2. TDA Topological Anomaly Windows (strict: both>2σ OR solo>3σ)")
    wl = shared['window_labels']
    output_lines.append(f"Window size: {WINDOW} weeks | Total windows computed: {len(wl)}")
    output_lines.append(f"Anomaly logic: (|z0|>{THRESHOLD} AND |z1|>{THRESHOLD}) OR |z0|>{STRONG_SOLO} OR |z1|>{STRONG_SOLO}")
    output_lines.append("")
    output_lines.append(f"{'window_end_week':<18} {'H0_z':>8} {'H1_z':>8}  trigger")
    for idx in shared['anomaly_idx']:
        trigger = []
        if shared['both_triggered'][idx]: trigger.append("both>2σ")
        if shared['strong_solo_h0'][idx]: trigger.append("H0-solo>3σ")
        if shared['strong_solo_h1'][idx]: trigger.append("H1-solo>3σ")
        output_lines.append(f"{wl[idx]:<18} {shared['z0'][idx]:>+8.2f} {shared['z1'][idx]:>+8.2f}  {', '.join(trigger)}")
    output_lines.append(f"\nTotal TDA anomaly windows: {len(shared['anomaly_idx'])}")

    subsection("A3. SARI/ILI Validation of TDA Anomaly Weeks")
    sari, ili = shared['sari'], shared['ili']
    rising, falling = shared['rising_weeks'], shared['falling_weeks']
    taw = shared['tda_anomaly_weeks']
    output_lines.append("Direction based on 3-week smoothed SARI trend at the anomaly week.")
    sari_rising = sari[[w for w in rising if w in sari.index]]
    sari_falling = sari[[w for w in falling if w in sari.index]]
    ili_rising = ili[[w for w in rising if w in ili.index]]
    ili_falling = ili[[w for w in falling if w in ili.index]]
    output_lines.append(f"SARI median (all weeks):          {sari.median():.2f}")
    output_lines.append(f"SARI median at rising anomalies:  {sari_rising.median():.2f}" if len(sari_rising) > 0 else "SARI median at rising anomalies:  n/a")
    output_lines.append(f"SARI median at falling anomalies: {sari_falling.median():.2f}" if len(sari_falling) > 0 else "SARI median at falling anomalies: n/a")
    output_lines.append(f"ILI median (all weeks):           {ili.median():.2f}")
    output_lines.append(f"ILI median at rising anomalies:   {ili_rising.median():.2f}" if len(ili_rising) > 0 else "ILI median at rising anomalies:   n/a")
    output_lines.append(f"ILI median at falling anomalies:  {ili_falling.median():.2f}" if len(ili_falling) > 0 else "ILI median at falling anomalies:  n/a")
    output_lines.append("")
    output_lines.append(f"{'week':<18} {'direction':<12} {'sari_value':>12} {'ili_value':>12}")
    for w in sorted(taw):
        direction = "RISING" if w in rising else "FALLING"
        sari_val = f"{sari[w]:.2f}" if w in sari.index else "n/a"
        ili_val = f"{ili[w]:.2f}" if w in ili.index else "n/a"
        output_lines.append(f"{w:<18} {direction:<12} {sari_val:>12} {ili_val:>12}")

    subsection("A4. Seasonal Pattern — ISO Week Distribution of TDA Anomalies")
    output_lines.append("Frequency of anomaly window end-weeks by ISO week number:")
    iso_weeks = [int(str(w).split('-W')[1]) if '-W' in str(w) else pd.Timestamp(w).isocalendar()[1] for w in taw]
    iso_counter = {}
    for w in iso_weeks:
        iso_counter[w] = iso_counter.get(w, 0) + 1
    for iso_w in sorted(iso_counter):
        output_lines.append(f"  ISO week {iso_w:>2}: ({iso_counter[iso_w]})")

    section("BLOCK B: CURRENT STATE (most recent observations)")
    subsection(f"B1. TDA Entropy Trajectory: Last {ENTROPY_TRAIL} Windows")
    output_lines.append("Read the direction of travel: is entropy climbing toward the anomaly threshold?")
    output_lines.append(f"Threshold: ±{THRESHOLD}σ (both) | ±{STRONG_SOLO}σ (solo)")
    output_lines.append("")
    output_lines.append(f"{'window_end_week':<18} {'H0_z':>8} {'H1_z':>8}  status")
    trail_indices = list(range(max(0, len(wl) - ENTROPY_TRAIL), len(wl)))
    for idx in trail_indices:
        status = "ANOMALY" if shared['anomaly_mask'][idx] else ("elevated" if abs(shared['z0'][idx]) > 1.5 or abs(shared['z1'][idx]) > 1.5 else "normal")
        output_lines.append(f"{wl[idx]:<18} {shared['z0'][idx]:>+8.2f} {shared['z1'][idx]:>+8.2f}  {status}")
    recent_z0 = shared['z0'][trail_indices]
    recent_z1 = shared['z1'][trail_indices]
    output_lines.append("")
    output_lines.append(f"H0 entropy trend over last {ENTROPY_TRAIL} windows: {'RISING' if recent_z0[-1] > recent_z0[0] else 'FALLING'}  ({recent_z0[0]:+.2f} → {recent_z0[-1]:+.2f})")
    output_lines.append(f"H1 entropy trend over last {ENTROPY_TRAIL} windows: {'RISING' if recent_z1[-1] > recent_z1[0] else 'FALLING'}  ({recent_z1[0]:+.2f} → {recent_z1[-1]:+.2f})")
    current_anomalous = shared['anomaly_mask'][trail_indices[-1]]
    output_lines.append(f"Most recent window anomalous: {'YES ' if current_anomalous else 'NO'}")

    subsection("B2. PCA Positioning: Last 6 Weeks vs Historical Centroid")
    pca, pca_dist = shared['pca'], shared['pca_dist']
    weeks, X_pca, pca_dist_z = shared['weeks'], shared['X_pca'], shared['pca_dist_z']
    output_lines.append(f"Variance explained by 2 PCs: {sum(pca.explained_variance_ratio_)*100:.1f}%")
    output_lines.append(f"Historical mean distance from centroid: {pca_dist.mean():.3f}  std: {pca_dist.std():.3f}")
    output_lines.append("")
    output_lines.append(f"{'week':<18} {'PC1':>8} {'PC2':>8} {'dist_centroid':>15} {'dist_z':>8}  position")
    for i in range(max(0, len(weeks) - 6), len(weeks)):
        w, pc1, pc2 = weeks[i], X_pca[i, 0], X_pca[i, 1]
        dist, dz = pca_dist[i], pca_dist_z[i]
        pos = "FAR from normal" if dz > 2.0 else ("elevated" if dz > 1.0 else "within normal range")
        output_lines.append(f"{w:<18} {pc1:>8.3f} {pc2:>8.3f} {dist:>15.3f} {dz:>+8.2f}  {pos}")

    subsection("B3. Holt-Winters: Most Recent Fitted Residual")
    if hw_unavailable:
        output_lines.append(f"Holt-Winters model could not be fitted for {COUNTRY}/{PATHOGEN}.")
        output_lines.append("  Risk assessment relies on TDA (Blocks A2, B1, B2) and cross-validation (A3).")
    else:
        s = hw['series']
        output_lines.append(f"Most recent observed week : {s.index[-1].date()}")
        output_lines.append(f"Observed positivity       : {s.values[-1]:.2f}%")
        output_lines.append(f"Holt-Winters fitted value : {hw['fitted'].values[-1]:.2f}%")
        output_lines.append(f"Residual                  : {hw['residuals'][-1]:+.2f}%")
        output_lines.append(f"Residual z-score          : {hw['residual_z'][-1]:+.2f}")
        output_lines.append(f"Model currently surprised : {'YES ' if abs(hw['residual_z'][-1]) > 2 else 'NO'}")

    section("BLOCK C — FUTURE OUTLOOK")
    subsection(f"C1. Holt-Winters Forecast: Next {FORECAST_WEEKS} Weeks")
    if hw_unavailable:
        output_lines.append(f"Holt-Winters model could not be fitted for {COUNTRY}/{PATHOGEN}.")
        output_lines.append("  Risk assessment relies on TDA (Blocks A2, B1, B2) and cross-validation (A3).")
    else:
        output_lines.append(f"Pathogen: {PATHOGEN} | Country: {COUNTRY}")
        output_lines.append(f"Risk flag threshold: >{hw['risk_threshold']:.2f}% (historical p{RISK_PERCENTILE})")
        output_lines.append("")
        output_lines.append(f"{'week':<14} {'forecast_%':>12} {'risk_flag'}")
        for date, val in zip(hw['forecast'].index, hw['forecast'].values):
            flag = "HIGH RISK " if val >= hw['risk_threshold'] else "normal"
            output_lines.append(f"{str(date.date()):<14} {val:>12.2f}  {flag}")

    subsection("C2. TDA Entropy Trend: Forward Signal")
    output_lines.append("TDA does not forecast directly. Read the recent entropy trajectory to assess risk.")
    lead_times = []
    for idx in shared['anomaly_idx']:
        for lookback in range(1, WINDOW + 1):
            prev_idx = idx - lookback
            if prev_idx >= 0 and abs(shared['z0'][prev_idx]) <= 1.5 and abs(shared['z1'][prev_idx]) <= 1.5:
                lead_times.append(lookback)
                break
    avg_lead = int(np.mean(lead_times)) if lead_times else 4
    output_lines.append(f"Historical avg lead time (from elevation >1.5σ to confirmed anomaly): ~{avg_lead} weeks")
    output_lines.append("")
    last_z0, last_z1 = shared['z0'][trail_indices[-1]], shared['z1'][trail_indices[-1]]
    if current_anomalous:
        output_lines.append("STATUS: System is currently IN an anomalous topological state.")
    elif abs(last_z0) > 1.5 or abs(last_z1) > 1.5:
        output_lines.append(f"STATUS: Entropy is ELEVATED. H0_z={last_z0:+.2f}  H1_z={last_z1:+.2f}")
        output_lines.append(f"If trend continues, anomaly could materialize in ~{avg_lead} weeks.")
    else:
        output_lines.append(f"STATUS: Entropy within normal range. H0_z={last_z0:+.2f}  H1_z={last_z1:+.2f}")

    subsection("C3. Convergence Flags — Weeks Where Both Methods Agree")
    if hw_unavailable:
        output_lines.append("HW unavailable: see reason above.")
    else:
        output_lines.append("Highest-confidence risk: Holt-Winters forecasts HIGH RISK AND TDA signals anomaly/elevation.")
        output_lines.append("")
        tda_currently_elevated = current_anomalous or abs(last_z0) > 1.5 or abs(last_z1) > 1.5
        output_lines.append(f"{'forecast_week':<16} {'hw_positivity':>14} {'tda_context':<35} {'confidence'}")
        for date, val in zip(hw['forecast'].index, hw['forecast'].values):
            hw_high = val >= hw['risk_threshold']
            weeks_ahead = list(hw['forecast'].index).index(date) + 1
            tda_context_flag = tda_currently_elevated and weeks_ahead <= avg_lead + 2
            if hw_high and tda_context_flag:
                confidence = "(3) HIGH CONVERGENCE"
            elif hw_high:
                confidence = "(2) HW only"
            elif tda_context_flag:
                confidence = "(1) TDA signal only"
            else:
                confidence = "(0) no signal"
            tda_desc = "TDA: elevated/anomaly -> risk window" if tda_context_flag else "TDA: normal"
            output_lines.append(f"{str(date.date()):<16} {val:>14.2f}  {tda_desc:<35} {confidence}")

    return output_lines


def extract_risk_assessments(df_pos=None, shared=None, dfs=None) -> list[dict]:
    """
    Extract RiskAssessment-compatible dicts from TDA/Holt-Winters analysis.
    Aggregates pathogen×country into country-level risk assessments.

    Returns list of dicts matching: country, risk_level, spread_likelihood,
    reasoning, recommended_disease_focus, twelve_week_forecast.
    """
    if dfs is None:
        dfs = load_data()
    if shared is None:
        shared = compute_shared_tda(dfs)
    if df_pos is None:
        df_pos = shared['df_pos']

    combos = df_pos.groupby(['pathogen', 'countryname']).size().reset_index()
    combos = list(combos[['pathogen', 'countryname']].itertuples(index=False))

    by_country: dict[str, list[dict]] = {}
    for pathogen, country in combos:
        hw = _fit_holt_winters(df_pos, pathogen, country)
        if hw['unavailable'] or hw.get('forecast') is None:
            forecast_vals = [0.0] * FORECAST_WEEKS
            risk_threshold = 0.0
            forecast_start = pd.Timestamp.now()
        else:
            forecast = hw['forecast']
            forecast_vals = [float(v) for v in forecast.values]
            while len(forecast_vals) < FORECAST_WEEKS:
                forecast_vals.append(0.0)
            forecast_vals = forecast_vals[:FORECAST_WEEKS]
            risk_threshold = hw['risk_threshold']
            forecast_start = forecast.index[0]

        n_risk_weeks = sum(1 for v in forecast_vals if v >= risk_threshold)
        max_forecast = max(forecast_vals) if forecast_vals else 0.0
        spread_likelihood = min(1.0, max_forecast / 100.0 if max_forecast > 0 else 0.0)

        if n_risk_weeks >= 4:
            risk_level = "CRITICAL"
        elif n_risk_weeks >= 2:
            risk_level = "HIGH"
        elif n_risk_weeks >= 1 or spread_likelihood >= 0.2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        iso_year, iso_week, _ = forecast_start.isocalendar()
        forecast_start_week = f"{iso_year}-W{iso_week:02d}"

        assessment = {
            "country": country,
            "risk_level": risk_level,
            "spread_likelihood": round(spread_likelihood, 2),
            "reasoning": (
                f"{pathogen} Holt-Winters forecast: {n_risk_weeks} of 12 weeks above p75 threshold ({risk_threshold:.1f}%). "
                f"Max forecast: {max_forecast:.1f}%. Spread likelihood derived from forecast profile."
            ),
            "recommended_disease_focus": [pathogen],
            "twelve_week_forecast": {
                "weekly_cases_per_100k": [round(v, 2) for v in forecast_vals],
                "forecast_start_week": forecast_start_week,
            },
        }
        by_country.setdefault(country, []).append(assessment)

    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    result = []
    for country, assessments in by_country.items():
        best = max(assessments, key=lambda a: (a["risk_level"] != "LOW", levels.index(a["risk_level"]) if a["risk_level"] in levels else 0, a["spread_likelihood"]))
        diseases = sorted(set(d for a in assessments for d in a["recommended_disease_focus"]))
        result.append({
            "country": country,
            "risk_level": best["risk_level"],
            "spread_likelihood": best["spread_likelihood"],
            "reasoning": best["reasoning"],
            "recommended_disease_focus": diseases,
            "twelve_week_forecast": best["twelve_week_forecast"],
        })
    return result


def run_llm_risk_analyzer(output_path: Path | None = None) -> str:
    """
    Run LLM risk analysis for every pathogen×country combination.
    Writes output to llm_risk_output.txt in module_1a.
    Returns the output file path.
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "llm_risk_output.txt"

    print("Loading data from ECDC GitHub...")
    dfs = load_data()

    print("Computing shared TDA/PCA/SARI/ILI...")
    shared = compute_shared_tda(dfs)
    df_pos = shared['df_pos']

    combos = df_pos.groupby(['pathogen', 'countryname']).size().reset_index()[['pathogen', 'countryname']]
    combos = list(combos.itertuples(index=False))
    print(f"Running analysis for {len(combos)} pathogen×country combinations...")

    all_lines = []
    for i, (pathogen, country) in enumerate(combos):
        print(f"  [{i+1}/{len(combos)}] {pathogen} / {country}")
        sep = [
            "",
            "=" * 80,
            f"PATHOGEN: {pathogen} | COUNTRY: {country}",
            "=" * 80,
        ]
        block = run_single_pathogen_country(df_pos, shared, pathogen, country)
        all_lines.extend(sep)
        all_lines.extend(block)

    full_output = "\n".join(all_lines)
    output_path.write_text(full_output)
    print(f"\nOutput saved to: {output_path}")
    risk_assessments = extract_risk_assessments(df_pos=df_pos, shared=shared, dfs=dfs)
    return str(output_path), risk_assessments

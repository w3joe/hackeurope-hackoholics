"""
LLM Risk CSV/JSONL Exporter for Module 1A.
Generates training records (one per pathogen×country) with enriched format
sufficient for downstream LLM to predict disease evolution over next 12 weeks.

Run as: uv run python module_1a/llm_risk_csv.py
   or:  uv run python -m module_1a.llm_risk_csv
   or:  uv run python scripts/run_module_1a_llm_risk_csv.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Import from module_1a.llm_risk_analyzer (backend)
# When run as script: path has project root, backend is sibling of llm_finetune
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from module_1a.llm_risk_analyzer import (
        ENTROPY_TRAIL,
        FORECAST_WEEKS,
        RISK_PERCENTILE,
        WINDOW,
        compute_shared_tda,
        load_data,
        _fit_holt_winters,
    )
except ImportError:
    from backend.module_1a.llm_risk_analyzer import (
        ENTROPY_TRAIL,
        FORECAST_WEEKS,
        RISK_PERCENTILE,
        WINDOW,
        compute_shared_tda,
        load_data,
        _fit_holt_winters,
    )

warnings.filterwarnings("ignore")

TRAILING_WEEKS = 6
SEP = "|"  # internal separator for list fields in CSV


def _week_str(ts: pd.Timestamp) -> str:
    """Convert datetime to ISO week string (e.g. 2026-W07)."""
    y, w, _ = ts.isocalendar()
    return f"{y}-W{w:02d}"


def _yearweek_to_ts(yw: str) -> pd.Timestamp:
    """Parse 2024-W01 to start-of-week Timestamp."""
    return pd.to_datetime(yw + "-1", format="%G-W%V-%u")


def _build_record(
    df_pos: pd.DataFrame,
    shared: dict,
    pathogen: str,
    country: str,
    end_date: pd.Timestamp | None = None,
) -> dict | None:
    """
    Build one training record for pathogen×country.
    If end_date is set, fit HW on data up to that date and add target_* (actual observed next 12w).
    Returns None if HW unavailable and we skip the record.
    """
    hw = _fit_holt_winters(df_pos, pathogen, country, end_date=end_date)
    if hw["unavailable"]:
        return None

    wl = shared["window_labels"]
    # TDA state: use last N windows ending on or before end_date (or use latest if no end_date)
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        last_valid_idx = -1
        for i, lbl in enumerate(wl):
            ts = _yearweek_to_ts(str(lbl))
            if ts <= end_ts:
                last_valid_idx = i
        if last_valid_idx < 0:
            return None
        trail_start = max(0, last_valid_idx - ENTROPY_TRAIL + 1)
        trail_indices = list(range(trail_start, last_valid_idx + 1))
    else:
        trail_indices = list(range(max(0, len(wl) - ENTROPY_TRAIL), len(wl)))
    recent_z0 = shared["z0"][trail_indices]
    recent_z1 = shared["z1"][trail_indices]
    last_z0 = float(recent_z0[-1])
    last_z1 = float(recent_z1[-1])
    current_anomalous = bool(shared["anomaly_mask"][trail_indices[-1]])

    # TDA status & trend
    if current_anomalous:
        tda_status = "anomaly"
    elif abs(last_z0) > 1.5 or abs(last_z1) > 1.5:
        tda_status = "elevated"
    else:
        tda_status = "normal"

    h0_trend = "rising" if recent_z0[-1] > recent_z0[0] else "falling"
    h1_trend = "rising" if recent_z1[-1] > recent_z1[0] else "falling"
    tda_trend = f"H0_{h0_trend},H1_{h1_trend}"

    # For historical mode, filter anomalies to those on or before end_date
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        anomaly_idx_filtered = [
            i for i in shared["anomaly_idx"]
            if _yearweek_to_ts(str(wl[i])) <= end_ts
        ]
        taw_filtered = {wl[i] for i in anomaly_idx_filtered}
        rising_filtered = shared["rising_weeks"] & taw_filtered
        falling_filtered = shared["falling_weeks"] & taw_filtered
    else:
        anomaly_idx_filtered = list(shared["anomaly_idx"])
        taw_filtered = shared["tda_anomaly_weeks"]
        rising_filtered = shared["rising_weeks"]
        falling_filtered = shared["falling_weeks"]

    # TDA lead time
    lead_times = []
    for idx in anomaly_idx_filtered:
        for lookback in range(1, WINDOW + 1):
            prev_idx = idx - lookback
            if prev_idx >= 0 and abs(shared["z0"][prev_idx]) <= 1.5 and abs(shared["z1"][prev_idx]) <= 1.5:
                lead_times.append(lookback)
                break
    lead_weeks = int(np.mean(lead_times)) if lead_times else 7

    # TDA past anomaly direction
    past_rising = sum(1 for w in taw_filtered if w in rising_filtered)
    past_falling = sum(1 for w in taw_filtered if w in falling_filtered)

    # TDA seasonal ISO weeks
    iso_weeks = []
    for w in taw_filtered:
        if isinstance(w, str) and "-W" in w:
            iso_weeks.append(int(w.split("-W")[1]))
        else:
            iso_weeks.append(pd.Timestamp(w).isocalendar()[1])
    seasonal_iso = sorted(set(iso_weeks)) if iso_weeks else []

    # Trailing observed (last 6 weeks)
    s = hw["series"]
    n_trail = min(TRAILING_WEEKS, len(s))
    trail_dates = s.index[-n_trail:]
    trail_weeks = [_week_str(d) for d in trail_dates]
    trail_values = [round(float(v), 2) for v in s.values[-n_trail:]]

    # Forecast (next 12 weeks)
    fc = hw["forecast"]
    fc_weeks = [_week_str(d) for d in fc.index]
    fc_values = [round(float(v), 2) for v in fc.values]
    high_risk_weeks = [
        _week_str(d)
        for d, v in zip(fc.index, fc.values)
        if v >= hw["risk_threshold"]
    ]

    # HW last observation
    last_obs = _week_str(s.index[-1])
    resid_z = float(hw["residual_z"][-1])
    surprised = abs(resid_z) > 2.0
    thr_pct = round(float(hw["risk_threshold"]), 2)

    # PCA (week at end of last trajectory window)
    weeks_list = shared["weeks"]
    pca_dist_z = shared["pca_dist_z"]
    idx_last = trail_indices[-1] + WINDOW - 1
    if idx_last >= len(weeks_list):
        idx_last = len(weeks_list) - 1
    dz = float(pca_dist_z[idx_last])
    if dz > 2.0:
        pca_position = "far"
    elif dz > 1.0:
        pca_position = "elevated"
    else:
        pca_position = "normal"

    # Target (actual observed next 12w) for historical training
    target_weeks = None
    target_values = None
    if end_date is not None and "full_series" in hw:
        full = hw["full_series"]
        fc_dates = list(fc.index)
        vals = []
        for d in fc_dates:
            if d in full.index:
                vals.append(round(float(full.loc[d]), 2))
            else:
                vals.append(None)
        non_nan = sum(1 for v in vals if v is not None)
        if non_nan >= 10:  # require 10+ actuals for valid training example
            target_weeks = fc_weeks
            target_values = [round(float(v), 2) if v is not None else None for v in vals]

    # Convergence
    tda_elevated = current_anomalous or abs(last_z0) > 1.5 or abs(last_z1) > 1.5
    hw_high_count = sum(1 for v in fc.values if v >= hw["risk_threshold"])
    tda_in_window = tda_elevated  # simplified: TDA signal present in risk window
    if hw_high_count > 0 and tda_in_window:
        convergence = 3
    elif hw_high_count > 0:
        convergence = 2
    elif tda_in_window:
        convergence = 1
    else:
        convergence = 0

    rec = {
        "pathogen": pathogen,
        "country": country,
        "trailing_weeks": trail_weeks,
        "trailing_values": trail_values,
        "forecast_hw_weeks": fc_weeks,
        "forecast_hw_values": fc_values,
        "forecast_hw_threshold_pct": thr_pct,
        "forecast_hw_high_risk_weeks": high_risk_weeks,
        "tda_n_anomalies": len(anomaly_idx_filtered),
        "tda_H0_z": round(last_z0, 2),
        "tda_H1_z": round(last_z1, 2),
        "tda_status": tda_status,
        "tda_trend": tda_trend,
        "tda_lead_weeks": lead_weeks,
        "tda_past_rising": past_rising,
        "tda_past_falling": past_falling,
        "tda_seasonal_iso_weeks": seasonal_iso,
        "hw_last_obs_week": last_obs,
        "hw_residual_z": round(resid_z, 2),
        "hw_surprised": surprised,
        "pca_dist_z": round(dz, 2),
        "pca_position": pca_position,
        "convergence": convergence,
    }
    if target_weeks is not None and target_values is not None:
        rec["target_weeks"] = target_weeks
        rec["target_values"] = target_values
        if end_date is not None:
            rec["cutoff_week"] = _week_str(pd.Timestamp(end_date))
    return rec


def _record_to_flat_dict(r: dict) -> dict:
    """Flatten record for CSV (lists as pipe-separated strings)."""
    out = {
        "pathogen": r["pathogen"],
        "country": r["country"],
        "trailing_weeks": SEP.join(r["trailing_weeks"]),
        "trailing_values": SEP.join(str(v) for v in r["trailing_values"]),
        "forecast_hw_weeks": SEP.join(r["forecast_hw_weeks"]),
        "forecast_hw_values": SEP.join(str(v) for v in r["forecast_hw_values"]),
        "forecast_hw_threshold_pct": r["forecast_hw_threshold_pct"],
        "forecast_hw_high_risk_weeks": SEP.join(r["forecast_hw_high_risk_weeks"]),
        "tda_n_anomalies": r["tda_n_anomalies"],
        "tda_H0_z": r["tda_H0_z"],
        "tda_H1_z": r["tda_H1_z"],
        "tda_status": r["tda_status"],
        "tda_trend": r["tda_trend"],
        "tda_lead_weeks": r["tda_lead_weeks"],
        "tda_past_rising": r["tda_past_rising"],
        "tda_past_falling": r["tda_past_falling"],
        "tda_seasonal_iso_weeks": SEP.join(str(w) for w in r["tda_seasonal_iso_weeks"]),
        "hw_last_obs_week": r["hw_last_obs_week"],
        "hw_residual_z": r["hw_residual_z"],
        "hw_surprised": r["hw_surprised"],
        "pca_dist_z": r["pca_dist_z"],
        "pca_position": r["pca_position"],
        "convergence": r["convergence"],
    }
    if "target_weeks" in r:
        out["target_weeks"] = SEP.join(r["target_weeks"])
        out["target_values"] = SEP.join(str(v) if v is not None else "" for v in r["target_values"])
        if "cutoff_week" in r:
            out["cutoff_week"] = r["cutoff_week"]
    return out


def _get_valid_end_dates(
    df_pos: pd.DataFrame,
    pathogen: str,
    country: str,
    step_weeks: int = 2,
) -> list[pd.Timestamp]:
    """Return end dates where we have 6w trailing + 12w future actuals for training."""
    sub = df_pos[
        (df_pos["pathogen"] == pathogen) & (df_pos["countryname"] == country)
    ][["yearweek"]].drop_duplicates().sort_values("yearweek")
    if len(sub) < 19:
        return []
    dates = pd.to_datetime(
        [f"{yw}-1" for yw in sub["yearweek"]],
        format="%G-W%V-%u",
    )
    first_ok = dates[5]   # need 6 weeks before (indices 0..5)
    last_ok = dates[-13]  # need 12 weeks after (indices -12..-1)
    if first_ok >= last_ok:
        return []
    out = []
    step = pd.Timedelta(weeks=step_weeks)
    d = pd.Timestamp(first_ok)
    end = pd.Timestamp(last_ok)
    while d <= end:
        out.append(d)
        d += step
    return out


def build_epi_prompt(record: dict) -> str:
    """Build epidemiological instruction prompt from flat record (same format as training)."""
    high_risk_weeks = (record.get("forecast_hw_high_risk_weeks") or "").split(SEP)
    seasonal_iso_weeks = (record.get("tda_seasonal_iso_weeks") or "").split(SEP)
    seasonal_iso_weeks = [int(x) for x in seasonal_iso_weeks if x.strip().isdigit()]

    trailing_weeks = (record.get("trailing_weeks") or "").split(SEP)
    trailing_values = []
    for x in (record.get("trailing_values") or "").split(SEP):
        try:
            trailing_values.append(float(x))
        except ValueError:
            trailing_values.append(0.0)
    forecast_hw_values = []
    for x in (record.get("forecast_hw_values") or "").split(SEP):
        try:
            forecast_hw_values.append(float(x))
        except ValueError:
            forecast_hw_values.append(0.0)

    prompt = f"""Analyze the following epidemiological data and provide a risk assessment:

**Pathogen:** {record.get('pathogen', '')}
**Country:** {record.get('country', '')}

**Recent Observation (Last 6 Weeks):**
Weeks: {', '.join(trailing_weeks)}
Cases per 100k: {', '.join(str(v) for v in trailing_values)}

**Statistical Forecast Signals:**
- Holt-Winters forecast (next 12w): {', '.join(f'{v:.1f}' for v in forecast_hw_values[:6])}... (truncated)
- Risk threshold: {record.get('forecast_hw_threshold_pct', 0)}% per 100k
- High-risk weeks predicted: {len(high_risk_weeks)} weeks

**Anomaly Detection (TDA):**
- Status: {record.get('tda_status', 'normal')}
- Trend: {record.get('tda_trend', '')}
- Number of anomalies: {record.get('tda_n_anomalies', 0)}
- Lead time: {record.get('tda_lead_weeks', 0)} weeks
- H0 z-score: {record.get('tda_H0_z', 0)}, H1 z-score: {record.get('tda_H1_z', 0)}

**Seasonality:**
- Historical peak weeks (ISO): {', '.join(f'W{w}' for w in seasonal_iso_weeks[:5])}{'...' if len(seasonal_iso_weeks) > 5 else ''}

**Pattern Analysis:**
- Last observation week: {record.get('hw_last_obs_week', '')}
- Residual z-score: {record.get('hw_residual_z', 0)}
- Forecast surprised by actual: {record.get('hw_surprised', False)}
- PCA distance z-score: {record.get('pca_dist_z', 0)}
- PCA position: {record.get('pca_position', 'normal')}
- Signal convergence: {record.get('convergence', 0)}/3

Provide a comprehensive 12-week forecast with risk assessment."""
    return prompt


def get_inference_records(
    df_pos: pd.DataFrame = None,
    shared: dict = None,
    dfs: dict = None,
) -> list[dict]:
    """
    Get records for inference (current snapshot, one per pathogen×country).
    Returns list of flat dicts suitable for building LLM prompts.
    Used by Module 1A when using local fine-tuned model.
    """
    if dfs is None:
        dfs = load_data()
    if shared is None:
        shared = compute_shared_tda(dfs)
    if df_pos is None:
        df_pos = shared["df_pos"]

    combos = (
        df_pos.groupby(["pathogen", "countryname"])
        .size()
        .reset_index()[["pathogen", "countryname"]]
    )
    combos = list(combos.itertuples(index=False))

    records: list[dict] = []
    for pathogen, country in combos:
        r = _build_record(df_pos, shared, pathogen, country)
        if r is not None:
            records.append(_record_to_flat_dict(r))
    return records


def run_llm_risk_csv(
    output_dir: Path | None = None,
    csv_name: str = "llm_risk_training.csv",
    jsonl_name: str = "llm_risk_training.jsonl",
    expand_historical: bool = True,
    step_weeks: int = 2,
) -> tuple[str, str]:
    """
    Generate CSV and JSONL training files for every pathogen×country.
    If expand_historical=True (default), generate many records per combo using past dates
    (trailing + forecast + actual target for next 12w). Step every step_weeks.
    Returns (csv_path, jsonl_path).
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    print("Loading data from ECDC GitHub...")
    dfs = load_data()

    print("Computing shared TDA/PCA/SARI/ILI...")
    shared = compute_shared_tda(dfs)
    df_pos = shared["df_pos"]

    combos = (
        df_pos.groupby(["pathogen", "countryname"])
        .size()
        .reset_index()[["pathogen", "countryname"]]
    )
    combos = list(combos.itertuples(index=False))

    records: list[dict] = []
    total_tasks = 0
    if expand_historical:
        for pathogen, country in combos:
            valid = _get_valid_end_dates(df_pos, pathogen, country, step_weeks)
            total_tasks += len(valid) if valid else 1
        if total_tasks == 0:
            total_tasks = len(combos)
    else:
        total_tasks = len(combos)

    done = 0
    for i, (pathogen, country) in enumerate(combos):
        if expand_historical:
            end_dates = _get_valid_end_dates(df_pos, pathogen, country, step_weeks)
            if not end_dates:
                r = _build_record(df_pos, shared, pathogen, country)
                if r is not None:
                    records.append(r)
                done += 1
                continue
            for end_date in end_dates:
                done += 1
                if done % 100 == 0 or done == total_tasks:
                    print(f"  [{done}/{total_tasks}] {pathogen} / {country} @ {end_date.date()}")
                r = _build_record(df_pos, shared, pathogen, country, end_date=end_date)
                if r is not None and "target_weeks" in r:
                    records.append(r)
        else:
            done += 1
            print(f"  [{done}/{total_tasks}] {pathogen} / {country}")
            r = _build_record(df_pos, shared, pathogen, country)
            if r is not None:
                records.append(r)

    # CSV
    flat = [_record_to_flat_dict(r) for r in records]
    df = pd.DataFrame(flat)
    csv_path = output_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path} ({len(records)} records)")

    # JSONL (full structure, better for LLM fine-tuning)
    jsonl_path = output_dir / jsonl_name
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"JSONL saved: {jsonl_path} ({len(records)} records)")

    return str(csv_path), str(jsonl_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate LLM risk training CSV/JSONL")
    p.add_argument("--no-expand", action="store_true", help="Only current snapshot (73 records), no historical expansion")
    p.add_argument("--step-weeks", type=int, default=2, help="Step between historical dates (default: 2)")
    args = p.parse_args()
    run_llm_risk_csv(expand_historical=not args.no_expand, step_weeks=args.step_weeks)

"""FastAPI service: receives Module 1B output and pushes alerts to Supabase Realtime."""

import logging
import sys
import ssl
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Fix SSL certificate verification on macOS (uses certifi's CA bundle)
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from src.config import ALERTS_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
from src.alerts.push import push_alerts_to_supabase
from module_1a.llm_risk_analyzer import WINDOW, THRESHOLD, STRONG_SOLO
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for worker threads (macOS)
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache
import io
from sklearn.preprocessing import StandardScaler
from ripser import ripser as rips

# Plot style
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["figure.dpi"] = 100

BASE = "https://raw.githubusercontent.com/EU-ECDC/Respiratory_viruses_weekly_data/main/data/"

FILES = [
    "ILIARIRates.csv",
    "SARIRates.csv",
    "SARITestsDetectionsPositivity.csv",
    "activityFluTypeSubtype.csv",
    "nonSentinelSeverity.csv",
    "nonSentinelTestsDetections.csv",
    "sentinelTestsDetectionsPositivity.csv",
    "sequencingVolumeDetectablePrevalence.csv",
    "variants.csv",
]

dfs = {}
for file in FILES:
    try:
        dfs[file] = pd.read_csv(BASE + file)
        print(f"{file:<55} shape: {dfs[file].shape}")
    except Exception as e:
        print(f"{file:<55} ERROR: {e}")


@lru_cache(maxsize=1)
def get_known_values():
    sentinel = dfs["sentinelTestsDetectionsPositivity.csv"]
    countries = {
        str(c).strip().lower(): str(c).strip()
        for c in sentinel["countryname"].dropna().unique()
    }
    pathogens = {
        str(p).strip().lower(): str(p).strip()
        for p in sentinel["pathogen"].dropna().unique()
    }
    return countries, pathogens


def _persistence_entropy(dgm):
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if lifetimes.sum() == 0:
        return 0
    p = lifetimes / lifetimes.sum()
    return -np.sum(p * np.log(p + 1e-10))


@lru_cache(maxsize=256)
def get_validation_weeks(country: str, pathogen: str):
    countries, pathogens = get_known_values()
    country_key = country.strip().lower()
    pathogen_key = pathogen.strip().lower()

    country_aliases = {
        "czech republic": "czechia",
    }
    pathogen_aliases = {
        "srv": "rsv",
    }

    country_key = country_aliases.get(country_key, country_key)
    pathogen_key = pathogen_aliases.get(pathogen_key, pathogen_key)

    if country_key not in countries:
        raise HTTPException(404, f"Unknown country '{country}'")
    if pathogen_key not in pathogens:
        raise HTTPException(404, f"Unknown pathogen '{pathogen}'")

    country_name = countries[country_key]
    pathogen_name = pathogens[pathogen_key]

    sent = dfs["sentinelTestsDetectionsPositivity.csv"]
    sentinel_series = (
        sent[
            (sent["countryname"] == country_name)
            & (sent["pathogen"] == pathogen_name)
            & (sent["indicator"] == "positivity")
            & (sent["age"] == "total")
        ]
        .groupby("yearweek")["value"]
        .median()
    )

    ns = dfs["nonSentinelTestsDetections.csv"]
    detections = (
        ns[
            (ns["countryname"] == country_name)
            & (ns["pathogen"] == pathogen_name)
            & (ns["age"] == "total")
            & (ns["indicator"] == "detections")
        ]
        .groupby("yearweek")["value"]
        .median()
    )
    tests = (
        ns[
            (ns["countryname"] == country_name)
            & (ns["pathogen"] == pathogen_name)
            & (ns["age"] == "total")
            & (ns["indicator"] == "tests")
        ]
        .groupby("yearweek")["value"]
        .median()
    )

    nonsentinel_series = (detections / tests * 100).replace([np.inf, -np.inf], np.nan)

    tda_matrix = (
        pd.DataFrame(
            {
                "sentinel_positivity": sentinel_series,
                "nonsentinel_positivity": nonsentinel_series,
            }
        )
        .sort_index()
        .dropna(how="all")
    )

    if tda_matrix.empty:
        raise HTTPException(
            404,
            f"No positivity data for country='{country_name}', pathogen='{pathogen_name}'",
        )

    thresh = max(1, int(tda_matrix.shape[1] * 0.6))
    tda_matrix = tda_matrix.dropna(thresh=thresh)
    tda_matrix = tda_matrix.fillna(tda_matrix.median(numeric_only=True))

    if len(tda_matrix) < WINDOW:
        raise HTTPException(
            400,
            (
                f"Not enough weekly points for TDA (need at least {WINDOW}, "
                f"got {len(tda_matrix)}) for country='{country_name}', "
                f"pathogen='{pathogen_name}'"
            ),
        )

    X_scaled = StandardScaler().fit_transform(tda_matrix)
    weeks = tda_matrix.index.tolist()

    entropies_h0, entropies_h1, window_labels = [], [], []
    for i in range(len(X_scaled) - WINDOW + 1):
        window = X_scaled[i : i + WINDOW]
        dgms = rips(window, maxdim=1)["dgms"]
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
    tda_anomaly_weeks = set(window_labels[i] for i in anomaly_idx)

    sari_df = dfs["SARIRates.csv"]
    sari = (
        sari_df[(sari_df["age"] == "total") & (sari_df["countryname"] == country_name)]
        .groupby("yearweek")["value"]
        .median()
        .sort_index()
    )

    if sari.empty:
        raise HTTPException(404, f"No SARI data for country '{country_name}'")

    sari_trend = sari.rolling(3, center=True).mean().diff()
    rising_weeks = {
        w for w in tda_anomaly_weeks if w in sari_trend.index and sari_trend[w] > 0
    }
    falling_weeks = {
        w for w in tda_anomaly_weeks if w in sari_trend.index and sari_trend[w] <= 0
    }

    return country_name, pathogen_name, rising_weeks, falling_weeks


def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Validate X-API-Key header."""
    if not ALERTS_API_KEY:
        raise HTTPException(500, "ALERTS_API_KEY not configured")
    if x_api_key != ALERTS_API_KEY:
        raise HTTPException(401, "Invalid API key")
    return x_api_key


app = FastAPI(title="Alerts API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "ok",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
    }


@app.get("/chart/validation/{country}/{pathogen}")
def validation_chart(country: str, pathogen: str):
    # ── Validation: do SARI and ILI confirm TDA anomaly weeks? ─────────────────

    country_name, pathogen_name, rising_weeks, falling_weeks = get_validation_weeks(
        country, pathogen
    )

    sari_df = dfs["SARIRates.csv"]
    sari = (
        sari_df[(sari_df["age"] == "total") & (sari_df["countryname"] == country_name)]
        .groupby("yearweek")["value"]
        .median()
        .sort_index()
    )

    ili_df = dfs["ILIARIRates.csv"]
    ili = (
        ili_df[
            (ili_df["age"] == "total")
            & (ili_df["countryname"] == country_name)
            & (ili_df["indicator"] == "ILIconsultationrate")
        ]
        .groupby("yearweek")["value"]
        .median()
        .sort_index()
    )

    if sari.empty and ili.empty:
        raise HTTPException(
            404, f"No SARI or ILI data for country '{country_name}'"
        )

    # ── Helper to get x positions and y values ──────────────────────────────────
    def get_positions(series, weeks):
        idx_list = list(series.index)
        xs = [idx_list.index(w) for w in weeks if w in idx_list]
        ys = [series[w] for w in weeks if w in idx_list]
        return xs, ys

    # Build series list: SARI and ILI when available (ILI not available for all countries)
    series_list = []
    labels = []
    titles = []
    colors = ["steelblue", "darkorchid"]
    if not sari.empty:
        series_list.append(sari)
        labels.append("SARI rate")
        titles.append("SARI Hospitalization Rate")
    if not ili.empty:
        series_list.append(ili)
        labels.append("ILI consultation rate")
        titles.append("ILI Consultation Rate")

    charts_rendered = ", ".join(titles)
    logger.info(
        "Validation chart rendered: country=%s pathogen=%s charts=[%s]",
        country_name,
        pathogen_name,
        charts_rendered,
    )

    # ── Plot ─────────────────────────────────────────────────────────────────────
    n_axes = len(series_list)
    fig, axes = plt.subplots(n_axes, 1, figsize=(16, 9), sharex=False)
    if n_axes == 1:
        axes = [axes]

    for ax, series, color_line, label_line, title in zip(
        axes,
        series_list,
        colors[:n_axes],
        labels,
        titles,
    ):
        ax.plot(
            range(len(series)),
            series.values,
            color=color_line,
            linewidth=1.2,
            label=label_line,
        )

        rx, ry = get_positions(series, rising_weeks)
        ax.scatter(
            rx,
            ry,
            color="tomato",
            zorder=5,
            s=90,
            label="Rising anomaly — RESTOCK ↑",
            marker="^",
        )

        fx, fy = get_positions(series, falling_weeks)
        ax.scatter(
            fx,
            fy,
            color="seagreen",
            zorder=5,
            s=90,
            label="Falling anomaly — WIND DOWN ↓",
            marker="v",
        )

        ax.set_title(
            f"{title} — TDA anomalies classified (3-week smoothed trend)",
            fontweight="bold",
        )
        ax.set_ylabel("Rate per 100k")
        ax.legend()

        step = max(1, len(series) // 15)
        ax.set_xticks(range(0, len(series), step))
        ax.set_xticklabels(
            list(series.index)[::step], rotation=45, ha="right", fontsize=8
        )

    plt.suptitle(
        (
            "Validation: Rising ↑ = restock signal | Falling ↓ = wind down signal\n"
            f"Country={country_name} | Pathogen={pathogen_name} "
            "(direction based on 3-week smoothed SARI trend)"
        ),
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/internal/alerts")
def post_alerts(
    body: dict,
    _: str = Depends(require_api_key),
):
    """
    Accept Module 1B output and push alerts to Supabase Realtime.

    Body: { "risk_assessments": [...] } (Module 1B output format)
    """
    risk_assessments = body.get("risk_assessments", [])
    if not risk_assessments and "error" in body:
        return {"ok": False, "message": "Module 1B error", "error": body.get("error")}

    result = push_alerts_to_supabase(risk_assessments)
    if result is None:
        raise HTTPException(
            503,
            "Supabase not configured or insert failed. Set SUPABASE_URL and SUPABASE_SERVICE_KEY.",
        )
    return {"ok": True, "alerts_count": len(result)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

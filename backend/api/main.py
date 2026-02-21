"""FastAPI service: receives Module 1B output and pushes alerts to Supabase Realtime."""

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.config import ALERTS_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
from src.alerts.push import push_alerts_to_supabase


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
    return {"status": "ok", "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)}


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

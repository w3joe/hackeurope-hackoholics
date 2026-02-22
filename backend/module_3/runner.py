"""Module 3 — Logistics Routing Engine.

Retrieves closest distributors from Supabase and builds routing plan per pharmacy.
"""

import math


def _get_supabase_client():
    """Get Supabase client. Returns None if not configured."""
    try:
        from src.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
        from supabase import create_client
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            return None
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except (ImportError, AttributeError):
        return None


def _fetch_distributors(supabase) -> list[dict]:
    """Fetch all distributors from Supabase."""
    if supabase is None:
        return []
    try:
        result = supabase.table("distributors").select("*").execute()
        return result.data or []
    except Exception:
        return []


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km."""
    R = 6371  # Earth radius km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _distributor_distance(pharmacy: dict, distributor: dict) -> float:
    """Compute distance from pharmacy to distributor. Uses lat/lng if available, else country/city heuristic."""
    ph_country = (pharmacy.get("country") or "").strip()
    ph_location = (pharmacy.get("location") or "").strip()
    ph_city = ph_location.split(",")[0].strip() if ph_location else ""
    dist_country = (distributor.get("country") or "").strip()
    dist_city = (distributor.get("city") or "").strip()
    ph_lat = pharmacy.get("latitude")
    ph_lon = pharmacy.get("longitude")
    d_lat = distributor.get("latitude")
    d_lon = distributor.get("longitude")

    if ph_lat is not None and ph_lon is not None and d_lat is not None and d_lon is not None:
        return _haversine_km(float(ph_lat), float(ph_lon), float(d_lat), float(d_lon))

    # Heuristic: same city = 15km, same country = 150km, different country = 500km
    if ph_country and dist_country:
        if ph_city and dist_city and ph_city.lower() == dist_city.lower():
            return 15.0
        if ph_country.lower() == dist_country.lower():
            return 150.0
    return 500.0


def _closest_distributors(
    pharmacy: dict,
    distributors: list[dict],
    limit: int = 5,
) -> list[dict]:
    """Return closest distributors for a pharmacy, sorted by distance."""
    if not distributors:
        return []
    with_distance = [
        {**d, "_distance_km": _distributor_distance(pharmacy, d)}
        for d in distributors
    ]
    sorted_d = sorted(with_distance, key=lambda x: x["_distance_km"])
    result = []
    for i, d in enumerate(sorted_d[:limit]):
        dist_km = d.pop("_distance_km", 0)
        result.append({
            "distributor_id": d.get("id", ""),
            "distributor_name": d.get("name", ""),
            "distance_km": round(dist_km, 1),
            "can_fulfil_by_urgency": i < 3,  # Top 3 can fulfil urgent
        })
    return result


def _fallback_distributors() -> list[dict]:
    """Fallback distributors when Supabase is not configured or empty."""
    return [
        {"id": "DIST-SANOFI", "name": "Sanofi Vaccines", "country": "France", "city": "Lyon", "latitude": 45.764, "longitude": 4.8357},
        {"id": "DIST-PFIZER", "name": "Pfizer/BioNTech", "country": "Germany", "city": "Berlin", "latitude": 52.52, "longitude": 13.405},
        {"id": "DIST-GSK", "name": "GSK Vaccines", "country": "Belgium", "city": "Wavre", "latitude": 50.7167, "longitude": 4.6167},
        {"id": "DIST-MODERNA", "name": "Moderna", "country": "Switzerland", "city": "Basel", "latitude": 47.5596, "longitude": 7.5886},
        {"id": "DIST-MSD", "name": "MSD", "country": "Netherlands", "city": "Haarlem", "latitude": 52.3874, "longitude": 4.6462},
    ]


def run_module_3(gap_reports: list[dict] | None = None) -> dict:
    """
    Run Module 3: Logistics Routing Engine.

    Fetches distributors from Supabase and builds routing plan per pharmacy.
    Each pharmacy gets closest distributors by distance (lat/lng or country/city heuristic).
    Falls back to hardcoded distributors if Supabase is not configured.
    """
    supabase = _get_supabase_client()
    distributors = _fetch_distributors(supabase)
    if not distributors:
        distributors = _fallback_distributors()

    if not gap_reports:
        return {"routing_plan": []}

    routing_plan: list[dict] = []
    for i, report in enumerate(gap_reports):
        pharmacy_id = report.get("pharmacy_id", "")
        location = report.get("location", f"Location for {pharmacy_id}")
        country = report.get("country", "")

        pharmacy = {
            "pharmacy_id": pharmacy_id,
            "location": location,
            "country": country,
        }

        available_distributors = _closest_distributors(pharmacy, distributors, limit=5)

        urgency = 0.9 - (i * 0.15)
        if urgency < 0.3:
            urgency = 0.3

        routing_plan.append({
            "pharmacy_id": pharmacy_id,
            "location": location,
            "delivery_urgency_score": round(urgency, 2),
            "estimated_delivery_time_hours": 4 + i * 2,
            "capacity_remaining_units": 500 - i * 80,
            "available_distributors": available_distributors,
        })

    return {"routing_plan": routing_plan}

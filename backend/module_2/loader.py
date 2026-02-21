"""Load and transform vaccine stock CSV to Module 2 inventory schema."""

import csv
from datetime import datetime
from pathlib import Path

VACCINE_CSV_PATH = Path(__file__).resolve().parent / "vaccine_stock_dataset.csv"

# Map Target_Disease to broad category for risk matching
DISEASE_CATEGORY = {
    "Influenza": "influenza",
    "COVID-19": "respiratory infections",
    "RSV": "respiratory infections",
    "Pneumonia": "respiratory infections",
}

# Map Country to region_id (aligns with Module 1A R1/R2/R3)
COUNTRY_TO_REGION = {
    "Germany": "R1",
    "UK": "R1",
    "France": "R1",
    "Austria": "R2",
    "Italy": "R2",
    "Poland": "R2",
    "Czech Republic": "R2",
    "Hungary": "R3",
    "Slovakia": "R3",
    "Romania": "R3",
    "Croatia": "R3",
    "Slovenia": "R3",
    "Bulgaria": "R3",
    "Serbia": "R3",
}

# Default unit cost USD by vaccine type (approximate market rates)
DEFAULT_UNIT_COST_USD = {
    "Influenza": 15.0,
    "COVID-19": 25.0,
    "RSV": 220.0,  # Arexvy/Abrysvo are expensive
    "Pneumonia": 180.0,  # Prevnar 13 / Pneumovax 23
}

DEFAULT_LEAD_TIME_DAYS = 5


def _parse_date(s: str) -> datetime | None:
    """Parse date in YYYY-MM-DD or DD/MM/YY format."""
    s = s.strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _days_until_expiry(expiry_str: str, snapshot_str: str = "2026-02-21") -> int:
    """Compute days until expiry from snapshot date."""
    exp = _parse_date(expiry_str)
    snap = _parse_date(snapshot_str)
    if exp and snap:
        delta = (exp - snap).days
        return max(0, delta)
    return 365  # default if parse fails


def load_vaccine_inventory(
    csv_path: Path | None = None,
) -> dict:
    """
    Load vaccine_stock_dataset.csv and transform to Module 2 inventory schema.

    Groups rows by Store_ID into pharmacies, each with stock items per Vaccine_Brand.
    CSV schema: Snapshot_Date, Country, City, Address, Postal_Code, Store_ID,
    Target_Disease, Vaccine_Brand, Manufacturer, Stock_Quantity, Min_Stock_Level,
    Expiry_Date, Storage_Type.
    """
    path = csv_path or VACCINE_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Vaccine CSV not found: {path}")

    pharmacies: dict[str, dict] = {}  # key: Store_ID

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        snapshot_date = "2026-02-21"
        for row in reader:
            store_id = row.get("Store_ID", "").strip()
            if not store_id:
                continue
            country = row.get("Country", "")
            city = row.get("City", "")
            address = row.get("Address", "")

            if store_id not in pharmacies:
                pharmacies[store_id] = {
                    "pharmacy_id": store_id,
                    "pharmacy_name": (address or f"Store {store_id}")[:60],
                    "location": f"{city}, {country}",
                    "region_id": COUNTRY_TO_REGION.get(country, "R3"),
                    "stock": [],
                }

            target_disease = row.get("Target_Disease", "Influenza")
            vaccine_brand = row.get("Vaccine_Brand", "Unknown")
            manufacturer = row.get("Manufacturer", "Unknown")
            try:
                stock_qty = int(row.get("Stock_Quantity", 0) or 0)
                min_stock = int(row.get("Min_Stock_Level", 50) or 50)
            except (ValueError, TypeError):
                # Skip row if numeric fields fail (e.g. unquoted commas in Address)
                continue
            expiry_str = row.get("Expiry_Date", "")
            row_snapshot = row.get("Snapshot_Date", "").strip() or snapshot_date

            days_expiry = _days_until_expiry(expiry_str, row_snapshot)
            unit_cost = DEFAULT_UNIT_COST_USD.get(target_disease, 20.0)
            reorder_qty = max(min_stock * 2, 50)

            stock_item = {
                "drug_name": vaccine_brand,
                "category": DISEASE_CATEGORY.get(target_disease, "vaccines"),
                "quantity": stock_qty,
                "unit_price_usd": unit_cost,
                "reorder_threshold": min_stock,
                "reorder_quantity": reorder_qty,
                "days_until_expiry": days_expiry,
                "supplier_id": f"{store_id}-{vaccine_brand}"[:50].replace(" ", "-"),
                "supplier_name": manufacturer,
                "supplier_lead_time_days": DEFAULT_LEAD_TIME_DAYS,
                "supplier_unit_cost_usd": unit_cost,
            }
            pharmacies[store_id]["stock"].append(stock_item)

    return {"pharmacies": list(pharmacies.values())}

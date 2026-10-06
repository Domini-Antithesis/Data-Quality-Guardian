"""Generate the messy demo datasets shipped with the app.

Deterministic (seeded), and every flaw is introduced on purpose so the
detectors have a known ground truth to be measured against.

    python scripts/make_samples.py
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "dq_agent" / "resources" / "samples"

PRODUCTS = [
    ("Blue Widget", "Widgets"),
    ("Red Gadget", "Gadgets"),
    ("Smart Device", "Electronics"),
    ("Green Doohickey", "Widgets"),
    ("Steel Bracket", "Hardware"),
    ("Copper Fitting", "Hardware"),
    ("Mini Sensor", "Electronics"),
    ("Cable Set", "Accessories"),
]
CITIES = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Pune", "Hyderabad"]
REPS = ["Priya Sharma", "Rohan Mehta", "Ananya Iyer", "Vikram Singh", "Meera Nair"]


def build_sales_rows(rng: random.Random, n: int) -> list[dict]:
    """A sales export with the flaws a real export actually has."""
    rows = []
    start = datetime(2026, 1, 5)
    for i in range(n):
        name, category = rng.choice(PRODUCTS)
        rep = rng.choice(REPS)
        city = rng.choice(CITIES)
        qty = rng.randint(1, 40)
        unit = round(rng.uniform(120, 4500), 2)

        # Flaw 1: three different date formats.
        date = start + timedelta(days=i // 3)
        style = rng.random()
        if style < 0.55:
            date_str = date.strftime("%Y-%m-%d")
        elif style < 0.85:
            date_str = date.strftime("%d/%m/%Y")
        else:
            date_str = date.strftime("%d %B %Y")

        # Flaw 2: inconsistent casing and stray whitespace in text columns.
        if rng.random() < 0.18:
            city = city.upper()
        elif rng.random() < 0.12:
            city = city.lower()
        if rng.random() < 0.10:
            city = f"  {city} "
        if rng.random() < 0.08:
            rep = rep.lower()

        # Flaw 3: a numeric column stored as text with non-numeric stragglers.
        revenue = rng.choice(["N/A", "pending", "TBD", "-"]) if rng.random() < 0.06 else f"{qty * unit:,.2f}"

        # Flaw 4: missing values, concentrated in two columns.
        email = f"{rep.split()[0].lower()}.{rng.randint(1, 99)}@example.com"
        if rng.random() < 0.07:
            email = rng.choice(["not-an-email", "missing@", "@example.com", "n/a"])
        discount = "" if rng.random() < 0.22 else str(rng.choice([0, 5, 10, 15, 20]))
        notes = (
            ""
            if rng.random() < 0.63
            else rng.choice(["Repeat customer", "Bulk order", "Priority delivery", "Contract pricing"])
        )

        rows.append(
            {
                "OrderID": f"ORD-{10000 + i}",
                "OrderDate": date_str,
                "Product": name,
                "Category": category,
                "City": city,
                "SalesRep": rep,
                "CustomerEmail": email,
                "Quantity": str(qty),
                "UnitPrice": f"{unit:.2f}",
                "Revenue": revenue,
                "DiscountPct": discount,
                "Currency": "INR",  # Flaw 5: constant column
                "Notes": notes,
            }
        )

    # Flaw 6: outliers, a plausible extra-zero typo.
    for idx in rng.sample(range(n), 4):
        rows[idx]["UnitPrice"] = f"{float(rows[idx]['UnitPrice']) * 100:.2f}"

    # Flaw 7: exact duplicate rows.
    for idx in rng.sample(range(n), max(3, n // 25)):
        rows.append(dict(rows[idx]))
    rng.shuffle(rows)
    return rows


def build_clean_rows(rng: random.Random, n: int) -> list[dict]:
    """A tidy dataset, so the app can honestly report that nothing is wrong."""
    rows = []
    start = datetime(2026, 3, 1)
    for i in range(n):
        name, category = PRODUCTS[i % len(PRODUCTS)]
        rows.append(
            {
                "OrderID": f"CLN-{20000 + i}",
                "OrderDate": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
                "Product": name,
                "Category": category,
                "City": CITIES[i % len(CITIES)],
                "Quantity": str(rng.randint(1, 20)),
                "UnitPrice": f"{rng.uniform(200, 900):.2f}",
            }
        )
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    rng = random.Random(14)
    messy = build_sales_rows(rng, 240)
    write(OUT / "messy_sales_data.csv", messy)
    write(OUT / "clean_orders.csv", build_clean_rows(random.Random(7), 60))
    print(f"wrote {len(messy)} messy rows and 60 clean rows to {OUT}")


if __name__ == "__main__":
    main()

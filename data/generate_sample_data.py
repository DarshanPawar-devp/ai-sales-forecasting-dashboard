"""Generate data/sample_sales.csv (synthetic, 2 years of daily orders).

Run:  python data/generate_sample_data.py
Includes one clearly rising and one clearly declining product so the
trend features have something to find.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed seed: the same file is produced on every run, so results are reproducible.
rng = np.random.default_rng(42)
dates = pd.date_range("2024-10-01", "2026-09-30", freq="D")

# name: (category, base price, base daily units, yearly growth factor)
# A growth factor above 1 means the product's sales rise over the two years;
# below 1 means they fall.
PRODUCTS = {
    "Wireless Mouse": ("Electronics", 20, 14, 1.0),
    "Mechanical Keyboard": ("Electronics", 75, 6, 1.9),   # rising
    "USB-C Hub": ("Electronics", 35, 8, 1.0),
    "Desk Lamp": ("Home", 28, 9, 0.9),
    "Office Chair": ("Home", 160, 3, 1.0),
    "Notebook Pack": ("Stationery", 9, 25, 0.45),          # declining
    "Gel Pens": ("Stationery", 6, 30, 1.0),
    "Water Bottle": ("Lifestyle", 15, 12, 1.25),
}
REGIONS = ["North", "South", "East", "West"]


def build_rows() -> list[tuple]:
    """Simulate one row per (day, product, region) that had at least one sale."""
    rows = []
    for i, d in enumerate(dates):
        progress = i / (len(dates) - 1)                     # 0.0 on the first day, 1.0 on the last
        weekly = 1.25 if d.dayofweek >= 5 else 1.0          # weekends sell 25% more
        seasonal = 1 + 0.25 * np.sin(2 * np.pi * (d.dayofyear - 300) / 365)  # smooth yearly wave
        for name, (cat, price, base, growth) in PRODUCTS.items():
            trend = growth ** (progress * 2)                # compounds over the two years
            for region in REGIONS:
                # Poisson gives realistic randomness around the expected units; the /4 splits
                # demand across the four regions, and the floor keeps the rate above zero.
                units = rng.poisson(max(base * trend * weekly * seasonal / 4, 0.05))
                if units == 0:
                    continue                                # no sale, so no row
                unit_price = price * rng.uniform(0.95, 1.05)  # +/-5% price wobble
                rows.append((d.date().isoformat(), name, cat, region, units, round(units * unit_price, 2)))
    return rows


if __name__ == "__main__":
    df = pd.DataFrame(build_rows(),
                      columns=["Order_Date", "Product_Name", "Category", "Region", "Units_Sold", "Revenue"])
    out = Path(__file__).parent / "sample_sales.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")

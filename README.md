# 📈 AI Sales Forecasting Dashboard

An interactive Streamlit app that turns **any sales CSV** into a clean analytics dashboard with top-seller and
trend detection, plus an AI forecast of future sales. No fixed column names: you map your own columns in the sidebar.

> Evolved from a student script that predicted demand with `LinearRegression` on a fixed dataset
> (`sales forcasting system.py`) into a general-purpose, portfolio-ready application.

## Features

| Area | What you get |
|------|--------------|
| **Data upload** | CSV upload, preview, column types, missing values, duplicates, data-quality warnings |
| **Column mapping** | Pick date, sales, quantity, product, category and region columns (only date and sales are required) |
| **Safe cleaning** | Invalid dates dropped, currency symbols handled (`$1,200` → 1200), missing values filled/labelled, duplicates removed, with a cleaning log |
| **KPIs** | Total sales, total quantity, average sale, highest sale, growth rate, number of products |
| **Charts** | Sales over time (daily/weekly/monthly), category, region and product performance (interactive Plotly) |
| **Filters** | Date range, product, category, region |
| **Top and trending** | Top sellers by revenue and quantity, Rising / Stable / Declining label for every product, fastest-growing, declining, and "needs attention" lists |
| **Forecasting** | Overall and per-product forecast for 7, 30, 60 or 90 days, with confidence range, MAE and RMSE, forecast table and CSV download |
| **AI Insights** | Plain-language summary: best seller, trending item, declining item, expected future sales |
| **Graceful fallbacks** | No product column? Product analysis is skipped with an explanation. Too little history? No forecast is invented, and the app says what is needed |

## Project structure

```
ai-sales-forecasting-dashboard/
├── app.py                      # Streamlit UI (sidebar, tabs, charts)
├── assets/
│   └── style.css               # dark dashboard theme (cards, hero, badges, tabs, sidebar)
├── utils/
│   ├── ui.py                   # HTML helpers: KPI cards, forecast hero, side panels, icons
│   ├── data_processing.py      # loading, column guessing, quality checks, cleaning
│   ├── analytics.py            # KPIs, aggregations, product trends, insight text
│   └── forecasting.py          # Prophet / Holt-Winters / Ridge forecasting + evaluation
├── data/
│   ├── sample_sales.csv        # generated sample dataset (8 products, 2 years)
│   └── generate_sample_data.py # script that creates the sample file
├── .streamlit/config.toml      # theme and upload limit
├── requirements.txt
└── README.md
```

## Setup and run

Requires Python 3.9+.

```bash
# 1. (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. start the app
streamlit run app.py
```

The app opens at http://localhost:8501. Tick **Use the built-in sample data** in the sidebar to try it instantly.

To regenerate the sample file: `python data/generate_sample_data.py`.

### Optional: Prophet

Prophet gives the best forecasts but is a heavier install. If it is missing the app automatically falls back to
Holt-Winters or scikit-learn, so it is entirely optional:

```bash
pip install prophet
```

## CSV format

Column names can be anything. You only need a date column and a numeric sales/revenue column; the rest are optional.

| Order_Date | Product_Name | Category | Region | Units_Sold | Revenue |
|------------|--------------|----------|--------|-----------:|--------:|
| 2025-01-03 | Wireless Mouse | Electronics | North | 12 | 240.00 |
| 2025-01-03 | Desk Lamp | Home | West | 5 | 130.00 |
| 2025-01-04 | Gel Pens | Stationery | East | 30 | 180.00 |

The app guesses likely columns from their names, and you can change any guess in the sidebar.

## How the AI forecasting works

1. **Aggregate**: sales are summed per period. Daily, weekly or monthly granularity is detected from your dates.
2. **Pick a model** (best available first):
   - **Prophet**, if installed
   - **Holt-Winters exponential smoothing** (statsmodels), with seasonality when there are two full cycles
   - **Ridge regression** (scikit-learn) on a time trend plus yearly and weekly seasonality features
3. **Evaluate honestly**: the most recent part of the history is held out, forecast, and compared with what really
   happened. This gives **MAE** and **RMSE**. Then the model is retrained on all data for the real forecast.
4. **Refuse when unsure**: fewer than 30 daily (20 weekly, 12 monthly) data points, or mostly-zero history,
   means no forecast is shown and the app explains what more data is needed.

Product-level forecasts run the same pipeline on each selected product (up to 5 at a time).

### Trend labels

For every product, the selected date range is split in half and sales in the latter half are compared with
the first half. More than +10% is **Rising**, less than −10% is **Declining**, anything else is **Stable**.
"Needs attention" lists declining products with above-median revenue, or any product down 30% or more.

## Technologies

Python · Streamlit · Pandas · NumPy · Plotly · scikit-learn · statsmodels · Prophet (optional)

## Limitations and ideas for next steps

- Forecasts use only the sales history itself; no prices, promotions or holidays yet.
- Days with no records count as zero sales in daily forecasts.
- Ideas: holiday effects, extra regressors (price, promotions), model comparison, scheduled email reports.

## Deploying (no database needed)

The app is stateless: uploaded CSVs are processed in memory for the current session and never written to disk or
a database. That means deployment is just "host the code".

### Option 1: Streamlit Community Cloud (free, easiest)

1. Create a GitHub repository and push this folder to it (`app.py` at the repo root).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
3. Choose your repository and branch, set the main file to `app.py`, and click **Deploy**.
4. You get a public link such as `https://your-app-name.streamlit.app` to put on your resume or portfolio.

Keep `prophet` out of `requirements.txt` for hosting (it is slow to build). The app uses Holt-Winters instead.

### Option 2: Hugging Face Spaces

Create a new Space, choose the **Streamlit** SDK, and upload the project files.

### Option 3: Render / Railway / any server

Use `pip install -r requirements.txt` as the build command and
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0` as the start command.

## The look

A dark dashboard with indigo/violet accents: sidebar controls, a segmented tab bar, a large forecast card with
backtest accuracy, a "What's driving sales" panel and product movers with sparklines.

## Customising the look

All styling lives in `assets/style.css` (colours are CSS variables at the top of the file) and the reusable HTML
components (KPI cards, badges, insight cards) are in `utils/ui.py`. Theme colours for Streamlit's own widgets are in
`.streamlit/config.toml` (dark base).

## Bug fixes in this version

| Where | Problem | Fix |
|-------|---------|-----|
| `app.py` `fmt()` | Divided by 10,000 but labelled the result "K", so 98,000 showed as "9.8K" | Separate threshold and divisor per unit |
| `data_processing.clean_data` | Duplicates were removed after keeping only the mapped columns, so genuine orders with the same date and amount were deleted | De-duplicate complete raw rows before selecting columns |
| `data_processing.load_csv` | `utf-8` was tried before `utf-8-sig`, so a BOM stayed in the first column name | `utf-8-sig` first |
| `data_processing.guess_column` | `$1,200`-style text columns were never suggested as the sales column | Columns that are mostly parseable as numbers now count |
| `data_processing.clean_data` | Mixed or aware time zones crashed date parsing; the row-removal log could double count | Parse as UTC then drop the zone; count each reason separately |
| `forecasting.forecast_sales` | Monthly forecasts were trimmed from the month's first day, dropping a month (90 days gave 2 months) | Trim from the end of the last period |
| `analytics.product_trends` | Relied on pandas keeping the index name | `rename_axis("product")` |
| `app.py` charts | `use_container_width` is being removed from newer Streamlit | `show_chart()` picks the supported argument |

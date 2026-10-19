"""Sales forecasting: Prophet -> Exponential Smoothing -> scikit-learn fallback.

The module never invents predictions: if the history is too short, it returns
a clear message explaining what is missing instead of a forecast.
"""
from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Model libraries are noisy; keep the app's console readable.
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# Prophet and statsmodels are optional. Missing (or broken) libraries are not an
# error: the app simply falls back to the next model in the chain.
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:  # not installed or broken backend
    HAS_PROPHET = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# Granularity settings: pandas rule, days per period, seasonal length, minimum points
FREQ = {
    "D": {"rule": "D", "days": 1, "season": 7, "min_points": 30, "label": "daily"},
    "W": {"rule": "W", "days": 7, "season": 52, "min_points": 20, "label": "weekly"},
    "M": {"rule": "MS", "days": 30, "season": 12, "min_points": 12, "label": "monthly"},
}


def detect_frequency(dates: pd.Series) -> str:
    """Guess whether records are daily, weekly or monthly from date gaps."""
    unique = pd.Series(sorted(dates.dt.normalize().unique()))
    if len(unique) < 3:
        return "D"
    gap = unique.diff().dt.days.median()  # the typical distance between two dates
    if gap >= 25:
        return "M"
    if gap >= 6:
        return "W"
    return "D"


def build_series(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate sales to one value per period. Missing periods count as 0 sales."""
    s = df.set_index("date")["sales"].resample(FREQ[freq]["rule"]).sum()
    # "ds" / "y" are the column names Prophet expects; the other models reuse them.
    return s.reset_index().rename(columns={"date": "ds", "sales": "y"})


def check_enough_data(series: pd.DataFrame, freq: str) -> str | None:
    """Return an explanation if data is insufficient, else None."""
    cfg = FREQ[freq]
    if len(series) < cfg["min_points"]:
        return (f"Only {len(series)} {cfg['label']} data points are available. "
                f"At least {cfg['min_points']} are needed for a dependable forecast. "
                f"Please upload a longer sales history.")
    if (series["y"] > 0).sum() < max(8, cfg["min_points"] // 3):
        return ("Most periods have zero sales, so there is not enough signal to forecast. "
                "More complete sales history is needed.")
    return None


def _period_end(last_ds: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Last day covered by the final history period.

    Weekly periods are labelled by their END (a Sunday), so the label already is
    the last day. Daily periods are a single day. Monthly periods are labelled by
    their START, so the real last day is the end of that month.
    """
    if freq == "M":
        return last_ds + pd.offsets.MonthEnd(0)
    return last_ds


# ---------------------------------------------------------------- models
# Each model function has the same contract:
#   (train, periods, freq) -> (dates, prediction, lower_bound, upper_bound)
def _fit_predict_prophet(train, periods, freq):
    m = Prophet(weekly_seasonality=(freq == "D"),
                # Yearly seasonality needs about two years of history to be learnable.
                yearly_seasonality=(len(train) * FREQ[freq]["days"] >= 730),
                daily_seasonality=False)
    m.fit(train)
    fut = m.make_future_dataframe(periods=periods, freq=FREQ[freq]["rule"])
    p = m.predict(fut).tail(periods)  # keep only the future rows
    return p["ds"].values, p["yhat"].values, p["yhat_lower"].values, p["yhat_upper"].values


def _fit_predict_hw(train, periods, freq):
    season = FREQ[freq]["season"]
    use_season = len(train) >= 2 * season  # need at least two full seasonal cycles
    model = ExponentialSmoothing(
        train["y"].values.astype(float),
        trend="add", damped_trend=True,  # damped so the trend flattens instead of exploding
        seasonal="add" if use_season else None,
        seasonal_periods=season if use_season else None,
        initialization_method="estimated",
    ).fit()
    yhat = model.forecast(periods)
    # Holt-Winters has no built-in interval: approximate a 95% band from the in-sample errors.
    sigma = np.std(train["y"].values - model.fittedvalues)
    dates = pd.date_range(train["ds"].iloc[-1], periods=periods + 1, freq=FREQ[freq]["rule"])[1:]
    return dates.values, yhat, yhat - 1.96 * sigma, yhat + 1.96 * sigma


def _fit_predict_sklearn(train, periods, freq):
    def feats(ds, t0):
        """Features: linear time trend, yearly sine/cosine, and day-of-week dummies."""
        t = np.arange(len(ds)) + t0
        X = pd.DataFrame({"t": t})
        doy = ds.dt.dayofyear.values
        X["sin_y"], X["cos_y"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        if freq == "D":
            for d in range(6):  # 6 dummies (not 7) so Sunday is the baseline
                X[f"dow{d}"] = (ds.dt.dayofweek.values == d).astype(int)
        return X

    model = Ridge(alpha=1.0)
    X = feats(train["ds"], 0)
    # Scale the trend column so it is on a similar range to the other features.
    mu, sd = X["t"].mean(), X["t"].std() or 1
    X["t"] = (X["t"] - mu) / sd
    model.fit(X, train["y"])
    dates = pd.Series(pd.date_range(train["ds"].iloc[-1], periods=periods + 1, freq=FREQ[freq]["rule"])[1:])
    Xf = feats(dates, len(train))  # t continues from where the training data stopped
    Xf["t"] = (Xf["t"] - mu) / sd
    yhat = model.predict(Xf)
    sigma = np.std(train["y"] - model.predict(X))
    return dates.values, yhat, yhat - 1.96 * sigma, yhat + 1.96 * sigma


def _run(train, periods, freq):
    """Try models in priority order. Returns (name, results)."""
    attempts = []
    if HAS_PROPHET:
        attempts.append(("Prophet", _fit_predict_prophet))
    if HAS_STATSMODELS:
        attempts.append(("Exponential Smoothing (Holt-Winters)", _fit_predict_hw))
    attempts.append(("Ridge Regression (scikit-learn)", _fit_predict_sklearn))  # always available
    last_err = None
    for name, fn in attempts:
        try:
            return name, fn(train, periods, freq)
        except Exception as e:  # fall through to the next model
            last_err = e
    raise RuntimeError(f"All forecasting models failed: {last_err}")


# ---------------------------------------------------------------- public API
def forecast_sales(df: pd.DataFrame, horizon_days: int, freq: str | None = None) -> dict:
    """Forecast total sales for `horizon_days` into the future.

    Returns a dict with keys: ok, message, model, freq, history, forecast, metrics.
    """
    result = {"ok": False, "message": "", "model": None, "freq": None,
              "history": None, "forecast": None, "metrics": None}
    if df.empty:
        result["message"] = "No data to forecast after filtering."
        return result

    freq = freq or detect_frequency(df["date"])
    series = build_series(df, freq)
    result["freq"], result["history"] = freq, series

    # Refuse to forecast when the history is too short or mostly empty.
    problem = check_enough_data(series, freq)
    if problem:
        result["message"] = problem
        return result

    periods = max(1, math.ceil(horizon_days / FREQ[freq]["days"]))

    # Evaluation: hold out the last part of the history, forecast it, compare.
    holdout = min(periods, len(series) // 4)
    if len(series) - holdout >= FREQ[freq]["min_points"] and holdout >= 3:
        try:
            _, (_, pred, _, _) = _run(series.iloc[:-holdout], holdout, freq)
            actual = series["y"].iloc[-holdout:].values
            pred = np.clip(pred, 0, None)  # sales cannot be negative
            result["metrics"] = {
                "mae": mean_absolute_error(actual, pred),
                "rmse": math.sqrt(mean_squared_error(actual, pred)),
                "holdout": holdout,
            }
        except Exception:
            result["metrics"] = None  # a failed backtest should not block the forecast

    # Final model: retrain on ALL the data and predict the real future.
    try:
        name, (dates, yhat, lo, hi) = _run(series, periods, freq)
    except RuntimeError as e:
        result["message"] = str(e)
        return result

    fc = pd.DataFrame({"date": pd.to_datetime(dates),
                       "predicted_sales": np.clip(yhat, 0, None),
                       "lower_bound": np.clip(lo, 0, None),
                       "upper_bound": np.clip(hi, 0, None)})
    # Trim to the exact number of days requested (matters for weekly/monthly data).
    # FIX: measure from the END of the last history period. Monthly periods are
    # labelled by their first day, so the old code counted from the 1st of the last
    # month and cut off a month (60 days gave 1 month, 90 days gave 2).
    end = _period_end(series["ds"].iloc[-1], freq) + pd.Timedelta(days=horizon_days)
    fc = fc[fc["date"] <= end].reset_index(drop=True)
    if fc.empty:  # horizon shorter than one period: keep the first period
        fc = pd.DataFrame({"date": pd.to_datetime(dates[:1]), "predicted_sales": np.clip(yhat[:1], 0, None),
                           "lower_bound": np.clip(lo[:1], 0, None), "upper_bound": np.clip(hi[:1], 0, None)})

    result.update(ok=True, model=name, forecast=fc)
    return result


def forecast_products(df: pd.DataFrame, horizon_days: int, products: list[str], freq: str | None = None) -> dict:
    """Run `forecast_sales` for each selected product. Returns {product: result}."""
    # Detect the frequency once on ALL data so every product uses the same granularity,
    # even if one product on its own has sparse records.
    freq = freq or detect_frequency(df["date"])
    return {p: forecast_sales(df[df["product"] == p], horizon_days, freq) for p in products}


def summarize_forecast(result: dict) -> dict:
    """Headline numbers for the forecast card, from a successful forecast result.

    Keys: total, prev_total, change_pct, lower, upper, peak_date, peak_value,
    average, periods, unit, accuracy, accuracy_label, holdout.
    `change_pct` and `accuracy` are None when they cannot be worked out honestly.
    """
    f, hist, metrics = result["forecast"], result["history"], result["metrics"]
    total = float(f["predicted_sales"].sum())

    # Compare with the same number of periods just before the forecast starts.
    prev_total = float(hist["y"].tail(len(f)).sum())
    change = (total - prev_total) / prev_total * 100 if prev_total > 0 else None

    # Backtest accuracy = 100% minus the average miss relative to average sales.
    accuracy = label = None
    if metrics:
        recent_avg = float(hist["y"].tail(metrics["holdout"]).mean())
        if recent_avg > 0:
            accuracy = float(np.clip(1 - metrics["mae"] / recent_avg, 0, 1)) * 100
            label = "High" if accuracy >= 85 else "Medium" if accuracy >= 70 else "Low"

    peak = f.loc[f["predicted_sales"].idxmax()]
    return {
        "total": total,
        "prev_total": prev_total,
        "change_pct": change,
        # Sum of the per-period 95% bounds: a deliberately cautious, wide range.
        "lower": float(f["lower_bound"].sum()),
        "upper": float(f["upper_bound"].sum()),
        "peak_date": peak["date"],
        "peak_value": float(peak["predicted_sales"]),
        "average": float(f["predicted_sales"].mean()),
        "periods": len(f),
        "unit": FREQ[result["freq"]]["label"],
        "accuracy": accuracy,
        "accuracy_label": label,
        "holdout": metrics["holdout"] if metrics else None,
    }

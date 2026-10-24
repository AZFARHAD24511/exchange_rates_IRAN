"""Refresh the three PNG charts embedded in README.md using TGJU data.

This script is intentionally independent of Streamlit so GitHub Actions can run it
headlessly once per day.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import arabic_reshaper
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bidi.algorithm import get_display

ROOT = Path(__file__).resolve().parents[1]
API_BASE_URL = "https://api.tgju.org/v1/market/indicator/summary-table-data/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; exchange-rates-IRAN-chart-updater/1.0)"}

# Keep these names unchanged: README.md already references them.
CHARTS = {
    "levels": ROOT / "smple_trend_exchange_rate.png",
    "gap": ROOT / "smple_trend_exchange_rate_d.png",
    "gap_ratio": ROOT / "smple_trend_exchange_rate_p.png",
}

PRESIDENTS = [
    {"name": "روحانی", "start": "2013-08-04", "end": "2021-08-04", "color": "purple"},
    {"name": "رئیسی", "start": "2021-08-05", "end": "2025-08-05", "color": "green"},
    # End date retained from the existing project code; update it when the project defines a final date.
    {"name": "پزشکیان", "start": "2025-08-06", "end": "2033-08-06", "color": "turquoise"},
]


def rtl(text: str) -> str:
    """Shape Persian labels for Matplotlib's left-to-right text layout."""
    return get_display(arabic_reshaper.reshape(text))


def clean_price(value: object) -> float:
    text = str(value)
    for fragment in ('<span class="high" dir="ltr">', '<span class="low" dir="ltr">', '</span>'):
        text = text.replace(fragment, '')
    return float(text.replace(',', '').strip())


def fetch_series(indicator: str) -> pd.Series:
    url = f"{API_BASE_URL}{indicator}?period=all&mode=full&ts={int(time.time() * 1000)}"
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    records: list[tuple[datetime, float]] = []
    for row in rows:
        try:
            records.append((datetime.strptime(row[6], "%Y/%m/%d"), clean_price(row[0])))
        except (IndexError, TypeError, ValueError):
            continue
    if not records:
        raise RuntimeError(f"TGJU returned no usable observations for indicator {indicator!r}.")
    series = pd.Series(
        data=[price for _, price in records],
        index=pd.DatetimeIndex([date for date, _ in records], name="date"),
        name=indicator,
        dtype="float64",
    )
    return series[~series.index.duplicated(keep="last")].sort_index()


def add_presidential_markers(ax: plt.Axes, start: pd.Timestamp, end: pd.Timestamp) -> None:
    for president in PRESIDENTS:
        date = pd.Timestamp(president["start"])
        if start <= date <= end:
            ax.axvline(date, color="gray", linestyle="--", linewidth=1, alpha=0.75)
            ax.text(
                date, 0.98, rtl(president["name"]), rotation=90,
                transform=ax.get_xaxis_transform(), verticalalignment="top",
                fontsize=9, color=president["color"],
            )


def annotate_last(ax: plt.Axes, series: pd.Series, label: str, color: str, percent: bool = False) -> None:
    x = series.index[-1]
    y = float(series.iloc[-1])
    formatted = f"{y:,.2f}%" if percent else f"{y:,.0f}"
    ax.annotate(
        f"{rtl(label)}: {formatted}", xy=(x, y), xytext=(-12, 12),
        textcoords="offset points", ha="right",
        arrowprops={"arrowstyle": "->", "color": color},
        bbox={"boxstyle": "round", "fc": "yellow", "alpha": 0.45},
    )

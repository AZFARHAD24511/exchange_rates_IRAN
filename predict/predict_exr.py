
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from io import StringIO
from pytrends.request import TrendReq
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Page configuration
st.set_page_config(
    page_title="Free-Market USD Forecast",
    layout="centered"
)

st.title("📈 ARIMA: Free-Market USD Forecast")

st.markdown(
    """
    This application uses historical free-market USD exchange rates
    and Google Trends search-interest data to forecast the USD rate
    for the next two days.
    """
)

# GitHub URL for Google Trends data
GITHUB_TRENDS_CSV_URL = (
    "https://raw.githubusercontent.com/AZFARHAD24511/"
    "exchange_rates_IRAN/main/predict/google_trends_daily.csv"
)

KEYWORDS = ["خرید دلار", "فروش دلار", "دلار فردایی"]


# Load free-market USD exchange-rate data from TGJU
@st.cache_data(ttl=3600)
def load_usd_data():
    timestamp = int(datetime.now().timestamp() * 1000)

    url = (
        "https://api.tgju.org/v1/market/indicator/"
        f"summary-table-data/price_dollar_rl?"
        f"period=all&mode=full&ts={timestamp}"
    )

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30
    )
    response.raise_for_status()

    data = response.json().get("data", [])
    records = []

    for row in data:
        try:
            price_text = (
                str(row[0])
                .replace('<span class="high" dir="ltr">', "")
                .replace("</span>", "")
                .replace(",", "")
                .strip()
            )

            price = float(price_text)
            date = datetime.strptime(row[6], "%Y/%m/%d")

            records.append({
                "date": date,
                "price": price
            })

        except (ValueError, TypeError, IndexError):
            continue

    if not records:
        raise ValueError(
            "No valid USD exchange-rate records were retrieved."
        )

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.set_index("date").sort_index()

    return df


# Load Google Trends data from GitHub
@st.cache_data(ttl=3600)
def load_trends_csv():
    response = requests.get(
        GITHUB_TRENDS_CSV_URL,
        timeout=30
    )
    response.raise_for_status()

    df = pd.read_csv(
        StringIO(response.text),
        parse_dates=["date"]
    )

    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    return df


# Fetch missing Google Trends data
@st.cache_data(ttl=3600)
def fetch_missing_trends(missing_dates, geo="IR"):
    if len(missing_dates) == 0:
        return pd.DataFrame(index=missing_dates)

    pytrends = TrendReq(hl="fa", tz=330)
    trends_list = []

    start_date = missing_dates.min()
    end_date = missing_dates.max()

    timeframe = (
        f"{start_date.strftime('%Y-%m-%d')} "
        f"{end_date.strftime('%Y-%m-%d')}"
    )

    for keyword in KEYWORDS:
        try:
            pytrends.build_payload(
                [keyword],
                timeframe=timeframe,
                geo=geo
            )

            temp = pytrends.interest_over_time()

            if not temp.empty and keyword in temp.columns:
                trends_list.append(temp[keyword].rename(keyword))

        except Exception:
            continue

    if not trends_list:
        return pd.DataFrame(index=missing_dates)

    new_trends = pd.concat(trends_list, axis=1)
    new_trends = new_trends.reindex(missing_dates)

    # Normalize each keyword's values to a 0-100 scale
    for column in new_trends.columns:
        maximum = new_trends[column].max()

        if pd.notna(maximum) and maximum > 0:
            new_trends[column] = (
                new_trends[column] / maximum * 100
            )

    return new_trends


# Load data
try:
    with st.spinner("Loading exchange-rate and Google Trends data..."):
        usd_df = load_usd_data()
        trends_df = load_trends_csv()

except Exception as error:
    st.error(f"Unable to load the required data: {error}")
    st.stop()


# Restrict the dataset to the most recent two years
two_years_ago = pd.Timestamp.now().normalize() - pd.Timedelta(days=730)

usd_df = usd_df[usd_df.index >= two_years_ago]
trends_df = trends_df[trends_df.index >= two_years_ago]

if usd_df.empty:
    st.error("No USD exchange-rate data is available for the last two years.")
    st.stop()


# Retrieve missing Google Trends dates
missing_dates = usd_df.index.difference(trends_df.index)

if not missing_dates.empty:
    with st.spinner("Retrieving missing Google Trends observations..."):
        try:
            new_trends = fetch_missing_trends(missing_dates)

            if not new_trends.empty:
                trends_df = pd.concat(
                    [trends_df, new_trends]
                ).sort_index()

                trends_df = trends_df[
                    ~trends_df.index.duplicated(keep="last")
                ]

        except Exception as error:
            st.warning(
                f"Some Google Trends data could not be retrieved: {error}"
            )


# Align the datasets by date
df = usd_df.join(trends_df, how="left").sort_index()

# Fill missing trend values only; do not fill missing USD prices
trend_columns = [
    column for column in df.columns if column != "price"
]

if trend_columns:
    df[trend_columns] = df[trend_columns].ffill().bfill()

# Keep observations with valid exchange-rate values
df = df.dropna(subset=["price"])

if len(df) < 30:
    st.error(
        "There are not enough historical observations to fit a reliable model."
    )
    st.stop()

price_series = df["price"].astype(float)


# Fit the ARIMA model
try:
    with st.spinner("Selecting and fitting the ARIMA model..."):
        model = auto_arima(
            price_series,
            seasonal=False,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True
        )

        forecast = model.predict(n_periods=2)
        forecast_values = np.asarray(forecast, dtype=float)

except Exception as error:
    st.error(f"Unable to fit the ARIMA model: {error}")
    st.stop()


# Forecast dates
last_date = price_series.index[-1]

forecast_dates = [
    last_date + timedelta(days=1),
    last_date + timedelta(days=2)
]


# In-sample evaluation metrics
in_sample_predictions = np.asarray(
    model.predict_in_sample(),
    dtype=float
)

# Align predictions and actual values in case their lengths differ
evaluation_length = min(
    len(price_series),
    len(in_sample_predictions)
)

actual_values = price_series.iloc[-evaluation_length:].to_numpy()
predicted_values = in_sample_predictions[-evaluation_length:]

mae = mean_absolute_error(
    actual_values,
    predicted_values
)

mape = mean_absolute_percentage_error(
    actual_values,
    predicted_values
) * 100


# Extract model coefficient p-values, if available
try:
    pvalues = model.arima_res_.pvalues
    parameter_names = model.arima_res_.param_names

    pvalue_text = ", ".join(
        f"{name}: {value:.4f}"
        for name, value in zip(parameter_names, pvalues)
    )

except Exception:
    pvalue_text = "Not available"


# Display model information
st.subheader("Model Performance")

col1, col2 = st.columns(2)

col1.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")
col2.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")

with st.expander("ARIMA model details"):
    st.write(f"Selected model: {model.order}")
    st.write(f"Last observed date: {last_date.date()}")
    st.write(f"Number of observations: {len(price_series)}")
    st.write(f"Model coefficient p-values: {pvalue_text}")

st.caption(
    "Note: MAE and MAPE above are calculated using in-sample predictions. "
    "They do not represent out-of-sample forecast accuracy."
)


# Historical data and two-day forecast chart
st.subheader("📊 Historical Data & 2-Day Forecast")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    price_series.index,
    price_series.values,
    label="Historical Free-Market USD Rate",
    linewidth=1.8
)

ax.axvline(
    last_date,
    color="gray",
    linestyle="--",
    label="Forecast Start"
)

ax.plot(
    forecast_dates,
    forecast_values,
    color="red",
    linestyle="--",
    marker="o",
    label="ARIMA Forecast"
)

for day_number, (forecast_date, forecast_value) in enumerate(
    zip(forecast_dates, forecast_values),
    start=1
):
    ax.annotate(
        f"Day +{day_number}: {forecast_value:,.0f}",
        xy=(forecast_date, forecast_value),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        arrowprops=dict(
            arrowstyle="->",
            color="red"
        )
    )

ax.set_title("Free-Market USD Exchange Rate: Historical Data and Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Exchange Rate (Source-Reported Units per USD)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Explanation of free-market and NIMA USD rates
st.caption(
    "**Free-Market USD Rate:** The US dollar exchange rate quoted in "
    "Iran's free market, outside the NIMA-specific transaction system. "
    "This is the rate modeled and forecast by this application. "
    "**NIMA USD Rate:** The rate associated with transactions reported "
    "through Iran's NIMA currency-trading system. It can differ from the "
    "free-market rate because the two rates apply to different transaction "
    "contexts and market arrangements. NIMA data are not included in this "
    "forecast chart."
)


# Display forecasts
st.subheader("🔮 Two-Day Forecast")

forecast_col1, forecast_col2 = st.columns(2)

forecast_col1.metric(
    label=f"Forecast for {forecast_dates[0].date()}",
    value=f"{forecast_values[0]:,.0f}"
)

forecast_col2.metric(
    label=f"Forecast for {forecast_dates[1].date()}",
    value=f"{forecast_values[1]:,.0f}"
)

st.info(
    f"MAE: {mae:,.2f} | MAPE: {mape:.2f}%"
)

# Download the data used by the model
csv_data = df.to_csv(index=True).encode("utf-8")

st.download_button(
    label="Download Historical Data (CSV)",
    data=csv_data,
    file_name="usd_forecast_data.csv",
    mime="text/csv"
)

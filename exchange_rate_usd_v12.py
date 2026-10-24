
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ============================================================
# Streamlit page configuration
# ============================================================
st.set_page_config(
    page_title="Iran USD Exchange Rates: Free Market vs NIMA",
    layout="wide"
)

st.markdown("""
---
© 2025 Dr. Farhadi. All rights reserved.

This application was developed by **Dr. Farhadi**,
PhD in Economics (Econometrics) and MSc in Data Science
and Artificial Intelligence.
""")


# ============================================================
# Presidential periods
# Note: dates are approximate period boundaries.
# The 2025–2033 period is an illustrative assumption.
# ============================================================
presidents = [
    {
        "name": "Rouhani",
        "start": "2013-08-04",
        "end": "2021-08-04",
        "color": "purple"
    },
    {
        "name": "Raisi",
        "start": "2021-08-05",
        "end": "2025-08-05",
        "color": "green"
    },
    {
        "name": "Pezeshkian (assumed period)",
        "start": "2025-08-06",
        "end": "2033-08-06",
        "color": "teal"
    }
]

for president in presidents:
    president["start"] = datetime.strptime(
        president["start"], "%Y-%m-%d"
    )
    president["end"] = datetime.strptime(
        president["end"], "%Y-%m-%d"
    )


# ============================================================
# Currency data scraper
# ============================================================
class CurrencyScraper:

    currencies_dict = {
        "USD_Azad": "price_dollar_rl",
        "USD_Nima": "nima_buy_usd"
    }

    api_base_url = (
        "https://api.tgju.org/v1/market/"
        "indicator/summary-table-data/"
    )

    @staticmethod
    def clean_price(price_value):
        """Extract a numeric price from a string or HTML fragment."""

        if price_value is None:
            raise ValueError("Missing price value")

        price_text = str(price_value)

        # Remove HTML tags
        price_text = re.sub(r"<[^>]+>", "", price_text)

        # Remove commas and whitespace
        price_text = price_text.replace(",", "").strip()

        # Keep digits, decimal point, and minus sign
        price_text = re.sub(r"[^0-9.\-]", "", price_text)

        if not price_text:
            raise ValueError("Price contains no numeric value")

        return float(price_text)

    @staticmethod
    def scrape_currency(currency):
        """Download historical exchange-rate data from TGJU."""

        if currency not in CurrencyScraper.currencies_dict:
            raise ValueError(f"Unsupported currency: {currency}")

        code = CurrencyScraper.currencies_dict[currency]

        timestamp_ms = int(datetime.now().timestamp() * 1000)

        url = (
            f"{CurrencyScraper.api_base_url}{code}"
            f"?period=all&mode=full&ts={timestamp_ms}"
        )

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict) or "data" not in data:
            raise ValueError(
                f"Unexpected API response for {currency}"
            )

        records = []

        for row in data["data"]:
            try:
                price = CurrencyScraper.clean_price(row[0])
                date = datetime.strptime(
                    str(row[6]), "%Y/%m/%d"
                )

                records.append({
                    "date": date,
                    "price": price
                })

            except (ValueError, TypeError, IndexError):
                continue

        if not records:
            raise ValueError(
                f"No valid historical records found for {currency}"
            )

        df = pd.DataFrame(records)

        # Remove duplicate dates, retaining the last record
        df = df.drop_duplicates(
            subset=["date"],
            keep="last"
        )

        df = df.set_index("date").sort_index()

        return df

    @staticmethod
    def test_stationarity(series):
        """Return the ADF test statistic and p-value."""

        clean_series = series.dropna()

        if len(clean_series) < 10:
            raise ValueError(
                "Insufficient observations for the ADF test"
            )

        if clean_series.nunique() < 2:
            raise ValueError(
                "The series is constant; ADF test is not applicable"
            )

        result = adfuller(clean_series)

        return {
            "statistic": result[0],
            "p_value": result[1],
            "critical_values": result[4]
        }

    @staticmethod
    def johansen_test(df):
        """Run the Johansen cointegration test."""

        clean_df = df.dropna()

        if len(clean_df) < 10:
            raise ValueError(
                "Insufficient observations for the Johansen test"
            )

        if any(clean_df[col].nunique() < 2 for col in clean_df.columns):
            raise ValueError(
                "A constant series was found; "
                "the Johansen test is not applicable"
            )

        result = coint_johansen(
            clean_df,
            det_order=0,
            k_ar_diff=1
        )

        return result.lr1, result.cvt


# ============================================================
# Annotation helper
# ============================================================
def annotate_last(ax, x, y, label, color="black"):

    ax.annotate(
        f"{label}: {y:,.2f}",
        xy=(x, y),
        xytext=(10, 10),
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="->",
            color=color
        ),
        bbox=dict(
            boxstyle="round",
            fc="white",
            alpha=0.85
        ),
        fontsize=9,
        color=color
    )


# ============================================================
# Presidential period markers
# ============================================================
def add_presidential_periods(ax, data_index):

    data_start = data_index.min()
    data_end = data_index.max()

    for president in presidents:

        start_date = president["start"]

        if data_start <= start_date <= data_end:

            ax.axvline(
                start_date,
                color="gray",
                linestyle="--",
                linewidth=1
            )

            ax.text(
                start_date,
                0.98,
                president["name"],
                transform=ax.get_xaxis_transform(),
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=8,
                color=president["color"]
            )


# ============================================================
# Main Streamlit application
# ============================================================
st.title("Iran USD Exchange Rates: Free Market vs NIMA")

try:

    # --------------------------------------------------------
    # Download exchange-rate data
    # --------------------------------------------------------
    with st.spinner("Downloading historical exchange-rate data..."):

        scraper = CurrencyScraper()

        df_azad = scraper.scrape_currency("USD_Azad")
        df_nima = scraper.scrape_currency("USD_Nima")

    # Align both series by date
    combined_df = pd.merge(
        df_azad["price"],
        df_nima["price"],
        left_index=True,
        right_index=True,
        suffixes=("_azad", "_nima"),
        how="inner"
    )

    combined_df = combined_df.sort_index()

    if combined_df.empty:

        st.warning(
            "No overlapping observations were found for "
            "the free-market and NIMA exchange rates."
        )

        st.stop()

    # --------------------------------------------------------
    # Calculate exchange-rate gap and percentage
    # --------------------------------------------------------
    combined_df["price_gap"] = (
        combined_df["price_azad"] -
        combined_df["price_nima"]
    )

    combined_df["price_gap_ratio"] = (
        combined_df["price_gap"] /
        combined_df["price_azad"].replace(0, float("nan"))
    ) * 100

    # --------------------------------------------------------
    # Save and download CSV
    # --------------------------------------------------------
    today_str = datetime.now().strftime("%Y%m%d")

    os.makedirs("data", exist_ok=True)

    csv_path = f"data/usd_data_{today_str}.csv"

    combined_df.to_csv(
        csv_path,
        index=True
    )

    csv_data = combined_df.to_csv(
        index=True
    ).encode("utf-8")

    st.download_button(
        label="Download Processed Data (CSV)",
        data=csv_data,
        file_name=f"usd_data_{today_str}.csv",
        mime="text/csv"
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------
    st.subheader("Latest Exchange Rates")

    latest = combined_df.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Free-Market USD Rate",
        f"{latest['price_azad']:,.0f}"
    )

    col2.metric(
        "NIMA USD Rate",
        f"{latest['price_nima']:,.0f}"
    )

    col3.metric(
        "Exchange-Rate Gap",
        f"{latest['price_gap']:,.0f}"
    )

    st.caption(
        "Rates are displayed in the source data's reported unit. "
        "Verify whether the TGJU values represent rials or tomans "
        "before interpreting or comparing the absolute levels."
    )

    # --------------------------------------------------------
    # ADF stationarity tests
    # --------------------------------------------------------
    st.subheader("Augmented Dickey-Fuller (ADF) Stationarity Tests")

    adf_col1, adf_col2 = st.columns(2)

    try:

        adf_azad = scraper.test_stationarity(
            combined_df["price_azad"]
        )

        adf_col1.write("**Free-Market USD Rate**")
        adf_col1.write(
            f"ADF statistic: {adf_azad['statistic']:.4f}"
        )
        adf_col1.write(
            f"p-value: {adf_azad['p_value']:.4f}"
        )

        if adf_azad["p_value"] < 0.05:
            adf_col1.info(
                "Reject the unit-root null hypothesis at the 5% level."
            )
        else:
            adf_col1.info(
                "Fail to reject the unit-root null hypothesis "
                "at the 5% level."
            )

    except Exception as error:
        adf_col1.warning(
            f"Free-market ADF test could not be completed: {error}"
        )

    try:

        adf_nima = scraper.test_stationarity(
            combined_df["price_nima"]
        )

        adf_col2.write("**NIMA USD Rate**")
        adf_col2.write(
            f"ADF statistic: {adf_nima['statistic']:.4f}"
        )
        adf_col2.write(
            f"p-value: {adf_nima['p_value']:.4f}"
        )

        if adf_nima["p_value"] < 0.05:
            adf_col2.info(
                "Reject the unit-root null hypothesis at the 5% level."
            )
        else:
            adf_col2.info(
                "Fail to reject the unit-root null hypothesis "
                "at the 5% level."
            )

    except Exception as error:
        adf_col2.warning(
            f"NIMA ADF test could not be completed: {error}"
        )

    st.caption(
        "The ADF test evaluates the null hypothesis that a series "
        "has a unit root. Its result depends on the specification "
        "and should be interpreted alongside the economic context."
    )

    # --------------------------------------------------------
    # Johansen cointegration test
    # --------------------------------------------------------
    st.subheader("Johansen Cointegration Test")

    try:

        johansen_stat, johansen_crit = scraper.johansen_test(
            combined_df[["price_azad", "price_nima"]]
        )

        johansen_table = pd.DataFrame(
            {
                "Rank Null Hypothesis": [
                    "r = 0",
                    "r <= 1"
                ],
                "Trace Statistic": johansen_stat,
                "Critical Value (90%)": johansen_crit[:, 0],
                "Critical Value (95%)": johansen_crit[:, 1],
                "Critical Value (99%)": johansen_crit[:, 2]
            }
        )

        st.dataframe(
            johansen_table,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "For each rank null hypothesis, compare the trace "
            "statistic with the chosen critical value. The test "
            "specification and integration properties of the series "
            "should be considered when drawing conclusions."
        )

    except Exception as error:

        st.warning(
            f"Johansen cointegration test could not be completed: {error}"
        )

    # ========================================================
    # Chart 1: Exchange rates
    # ========================================================
    st.subheader("Free-Market and NIMA USD Exchange Rates")

    fig1, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(
        combined_df.index,
        combined_df["price_azad"],
        label="Free-Market USD Rate",
        color="blue",
        linewidth=1.5
    )

    ax1.plot(
        combined_df.index,
        combined_df["price_nima"],
        label="NIMA USD Rate",
        color="orange",
        linewidth=1.5
    )

    add_presidential_periods(ax1, combined_df.index)

    annotate_last(
        ax1,
        combined_df.index[-1],
        combined_df["price_azad"].iloc[-1],
        "Free-Market USD",
        "blue"
    )

    annotate_last(
        ax1,
        combined_df.index[-1],
        combined_df["price_nima"].iloc[-1],
        "NIMA USD",
        "orange"
    )

    ax1.set_title(
        "Iran: Free-Market vs. NIMA USD Exchange Rates"
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Exchange Rate (Source-Reported Units per USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.caption(
        "**Definitions:** The Free-Market USD Rate is the US dollar "
        "price quoted in Iran's free market. The NIMA USD Rate refers "
        "to the rate reported for transactions through Iran's NIMA "
        "(Integrated System for Currency Transactions) platform. "
        "These rates represent different exchange-market arrangements "
        "and may differ. Historical NIMA rates should be interpreted "
        "in the context of the exchange-rate policies and market "
        "structure in effect at the time."
    )

    # ========================================================
    # Chart 2: Absolute exchange-rate gap
    # ========================================================
    st.subheader("Free-Market and NIMA USD Exchange-Rate Gap")

    fig2, ax2 = plt.subplots(figsize=(14, 6))

    ax2.plot(
        combined_df.index,
        combined_df["price_gap"],
        label="Exchange-Rate Gap",
        color="black",
        linewidth=1.5
    )

    add_presidential_periods(ax2, combined_df.index)

    annotate_last(
        ax2,
        combined_df.index[-1],
        combined_df["price_gap"].iloc[-1],
        "Latest Gap",
        "red"
    )

    ax2.axhline(
        0,
        color="gray",
        linewidth=0.8
    )

    ax2.set_title(
        "USD Exchange-Rate Gap: Free Market Minus NIMA"
    )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rate Difference (Source-Reported Units per USD)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        "**Calculation:** Exchange-Rate Gap = Free-Market USD Rate "
        "− NIMA USD Rate. A positive value means the free-market rate "
        "is higher than the NIMA rate; a negative value means it is lower."
    )

    # ========================================================
    # Chart 3: Percentage exchange-rate gap
    # ========================================================
    st.subheader(
        "Exchange-Rate Gap as a Percentage of the Free-Market Rate"
    )

    fig3, ax3 = plt.subplots(figsize=(14, 6))

    ax3.plot(
        combined_df.index,
        combined_df["price_gap_ratio"],
        label="Gap (% of Free-Market Rate)",
        color="green",
        linewidth=1.5
    )

    add_presidential_periods(ax3, combined_df.index)

    annotate_last(
        ax3,
        combined_df.index[-1],
        combined_df["price_gap_ratio"].iloc[-1],
        "Latest Gap",
        "green"
    )

    ax3.axhline(
        0,
        color="gray",
        linewidth=0.8
    )

    ax3.set_title(
        "USD Exchange-Rate Gap as a Percentage of the Free-Market Rate"
    )

    ax3.set_xlabel("Date")
    ax3.set_ylabel("Exchange-Rate Gap (%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "**Calculation:** Percentage Gap = "
        "(Free-Market USD Rate − NIMA USD Rate) "
        "/ Free-Market USD Rate × 100. "
        "This expresses the absolute rate difference as a percentage "
        "of the free-market rate."
    )

except requests.RequestException as error:

    st.error(
        "Could not retrieve exchange-rate data from TGJU. "
        f"Please check your internet connection and try again. Details: {error}"
    )

except Exception as error:

    st.error(
        f"An error occurred while running the application: {error}"
    )

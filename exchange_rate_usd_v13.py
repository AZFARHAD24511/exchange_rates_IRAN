
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ============================================================
# Streamlit page configuration
# ============================================================
st.set_page_config(
    page_title="Iran USD Exchange Rates: Free Market vs. NIMA",
    layout="wide"
)

st.markdown("""
---
This application was developed by **Dr. Farhadi**.

For more details, please visit the project's
[GitHub repository](https://github.com/AZFARHAD24511/exchange_rates_IRAN).
---
""")


# ============================================================
# Presidential periods
# Note: the 2025–2033 end date is an illustrative assumption.
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
        "color": "turquoise"
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
# Currency scraper
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
    def clean_price(price_str: str) -> float:
        """Remove HTML tags and thousands separators."""

        import re

        price_str = str(price_str)
        price_str = re.sub(r"<[^>]+>", "", price_str)
        price_str = price_str.replace(",", "").strip()
        price_str = re.sub(r"[^0-9.\-]", "", price_str)

        if not price_str:
            raise ValueError("Invalid price value")

        return float(price_str)

    @staticmethod
    def scrape_currency(currency: str) -> pd.DataFrame:
        """Retrieve historical exchange-rate observations from TGJU."""

        if currency not in CurrencyScraper.currencies_dict:
            raise ValueError(f"Unknown currency key: {currency}")

        code = CurrencyScraper.currencies_dict[currency]

        timestamp_ms = int(datetime.now().timestamp() * 1000)

        url = (
            f"{CurrencyScraper.api_base_url}{code}"
            f"?period=all&mode=full&ts={timestamp_ms}"
        )

        headers = {"User-Agent": "Mozilla/5.0"}

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
                f"No valid records found for {currency}"
            )

        df = pd.DataFrame(records)
        df = df.drop_duplicates(
            subset=["date"],
            keep="last"
        )
        df = df.set_index("date").sort_index()

        return df

    @staticmethod
    def test_stationarity(series):
        """Return the ADF p-value."""

        series = series.dropna()

        if len(series) < 10:
            raise ValueError(
                "Insufficient observations for the ADF test"
            )

        if series.nunique() < 2:
            raise ValueError(
                "The series is constant; ADF test cannot be applied"
            )

        return adfuller(series)[1]

    @staticmethod
    def johansen_test(df):
        """Return Johansen trace statistics and critical values."""

        df = df.dropna()

        if len(df) < 10:
            raise ValueError(
                "Insufficient observations for the Johansen test"
            )

        if any(df[col].nunique() < 2 for col in df.columns):
            raise ValueError(
                "A constant series was found; "
                "Johansen test cannot be applied"
            )

        result = coint_johansen(
            df,
            det_order=0,
            k_ar_diff=1
        )

        return result.lr1, result.cvt


# ============================================================
# Annotate the latest value on a chart
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
# Add presidential period markers to a chart
# ============================================================
def add_presidential_periods(ax, date_index):

    data_start = date_index.min()
    data_end = date_index.max()

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
# Main application
# ============================================================
st.title("Iran USD Exchange Rates: Free Market vs. NIMA")

try:

    # --------------------------------------------------------
    # Download historical data
    # --------------------------------------------------------
    with st.spinner("Downloading exchange-rate data from TGJU..."):

        scraper = CurrencyScraper()

        df_azad = scraper.scrape_currency("USD_Azad")
        df_nima = scraper.scrape_currency("USD_Nima")

    # --------------------------------------------------------
    # Merge both series by date
    # --------------------------------------------------------
    combined_df = pd.merge(
        df_azad["price"],
        df_nima["price"],
        left_index=True,
        right_index=True,
        suffixes=("_azad", "_nima"),
        how="inner"
    ).sort_index()

    if combined_df.empty:

        st.warning(
            "No overlapping observations were found for the "
            "free-market and NIMA exchange rates."
        )
        st.stop()

    # --------------------------------------------------------
    # Calculate the rate gap and percentage gap
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
    # Save and download processed CSV
    # --------------------------------------------------------
    today_str = datetime.now().strftime("%Y%m%d")

    os.makedirs("data", exist_ok=True)

    csv_path = f"data/usd_data_{today_str}.csv"

    combined_df.to_csv(csv_path, index=True)

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
    # Latest rates
    # --------------------------------------------------------
    st.subheader("Latest Exchange Rates")

    latest = combined_df.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Free-Market USD Rate",
        f"{latest['price_azad']:,.2f}"
    )

    col2.metric(
        "NIMA USD Rate",
        f"{latest['price_nima']:,.2f}"
    )

    col3.metric(
        "Exchange-Rate Gap",
        f"{latest['price_gap']:,.2f}"
    )

    st.caption(
        "The numerical unit reported by TGJU should be verified "
        "before interpreting the absolute rate levels. The chart "
        "labels intentionally avoid assuming rials or tomans."
    )

    # --------------------------------------------------------
    # ADF stationarity tests
    # --------------------------------------------------------
    st.subheader("Augmented Dickey-Fuller (ADF) Tests")

    adf_col1, adf_col2 = st.columns(2)

    try:
        p_azad = scraper.test_stationarity(
            combined_df["price_azad"]
        )

        adf_col1.write("**Free-Market USD Rate**")
        adf_col1.write(f"ADF p-value: {p_azad:.4f}")

        if p_azad < 0.05:
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
            f"Free-market ADF test failed: {error}"
        )

    try:
        p_nima = scraper.test_stationarity(
            combined_df["price_nima"]
        )

        adf_col2.write("**NIMA USD Rate**")
        adf_col2.write(f"ADF p-value: {p_nima:.4f}")

        if p_nima < 0.05:
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
            f"NIMA ADF test failed: {error}"
        )

    st.caption(
        "The ADF test evaluates the null hypothesis that a series "
        "has a unit root. Results depend on the model specification "
        "and should be interpreted in the economic context."
    )

    # --------------------------------------------------------
    # Johansen cointegration test
    # --------------------------------------------------------
    st.subheader("Johansen Cointegration Test")

    try:
        johansen_stat, johansen_crit = scraper.johansen_test(
            combined_df[["price_azad", "price_nima"]]
        )

        johansen_table = pd.DataFrame({
            "Rank Null Hypothesis": ["r = 0", "r <= 1"],
            "Trace Statistic": johansen_stat,
            "Critical Value (90%)": johansen_crit[:, 0],
            "Critical Value (95%)": johansen_crit[:, 1],
            "Critical Value (99%)": johansen_crit[:, 2]
        })

        st.dataframe(
            johansen_table,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Compare each trace statistic with its critical value "
            "to assess the corresponding rank null hypothesis. "
            "The results depend on the test specification."
        )

    except Exception as error:
        st.warning(
            f"Johansen cointegration test failed: {error}"
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
        "These rates reflect different exchange-market arrangements "
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

    ax2.axhline(0, color="gray", linewidth=0.8)

    ax2.set_title(
        "USD Exchange-Rate Gap: Free Market Minus NIMA"
    )
    ax2.set_xlabel("Date")
    ax2.set_ylabel(
        "Rate Difference (Source-Reported Units per USD)"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        "**Calculation:** Exchange-Rate Gap = Free-Market USD Rate "
        "− NIMA USD Rate. A positive value means the free-market rate "
        "is higher than the NIMA rate."
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

    ax3.axhline(0, color="gray", linewidth=0.8)

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
        "This expresses the rate difference as a percentage "
        "of the free-market rate."
    )

except requests.RequestException as error:
    st.error(
        "Could not retrieve data from TGJU. Check your internet "
        f"connection and try again. Details: {error}"
    )

except Exception as error:
    st.error(
        f"An error occurred while running the application: {error}"
    )

# streamlit_arima_predictor.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from io import StringIO
from pytrends.request import TrendReq
from pmdarima import auto_arima
import math

# پیکربندی صفحه
st.set_page_config(page_title="پیش‌بینی نرخ دلار آزاد", layout="centered")
st.title("📈 پیش‌بینی نرخ دلار آزاد با داده‌های گوگل ترندز و ARIMA")

# آدرس فایل ترندز در GitHub
GITHUB_TRENDS_CSV_URL = (
    'https://raw.githubusercontent.com/AZFARHAD24511/exchange_rates_IRAN/main/'
    'predict/google_trends_daily.csv'
)
KEYWORDS = ['خرید دلار', 'فروش دلار', 'دلار فردایی']

# بارگذاری داده‌های دلار آزاد از API
@st.cache_data(ttl=3600)
def load_usd_data():
    ts = int(datetime.now().timestamp() * 1000)
    url = (
        f"https://api.tgju.org/v1/market/indicator/"
        f"summary-table-data/price_dollar_rl?period=all&mode=full&ts={ts}"
    )
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = r.json().get('data', [])
    records = []
    for row in data:
        try:
            price = float(
                row[0]
                .replace(',', '')
                .replace('<span class="high" dir="ltr">', '')
                .replace('</span>', '')
            )
            date = datetime.strptime(row[6], "%Y/%m/%d")
            records.append({'date': date, 'price': price})
        except:
            continue
    df = pd.DataFrame(records)
    df = df.set_index('date').sort_index()
    return df

# بارگذاری داده‌های Google Trends از GitHub
@st.cache_data(ttl=3600)
def load_trends_csv():
    r = requests.get(GITHUB_TRENDS_CSV_URL)
    df = pd.read_csv(StringIO(r.text), parse_dates=['date'])
    return df.set_index('date').sort_index()

# تابع برای گرفتن داده‌های ناقص از Google Trends (برای missing dates)
@st.cache_data(ttl=3600)
def fetch_missing_trends(missing_dates, geo='IR'):
    pytrends = TrendReq(hl='fa', tz=330)
    df_all = []
    start = missing_dates.min()
    end = missing_dates.max()
    timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
    for keyword in KEYWORDS:
        pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
        tdf = pytrends.interest_over_time()
        if not tdf.empty:
            df_all.append(tdf[keyword].rename(keyword))
    if df_all:
        df_new = pd.concat(df_all, axis=1).loc[missing_dates]
        # نرمال‌سازی هر ستون به مقیاس 0-100
        return df_new.apply(lambda x: x / x.max() * 100)
    else:
        return pd.DataFrame(index=missing_dates)

# اجرای برنامه
with st.spinner("در حال بارگذاری داده‌ها..."):
    usd_df = load_usd_data()
    trends_df = load_trends_csv()

# استفاده از 2 سال اخیر
two_years_ago = datetime.now() - timedelta(days=730)
usd_df = usd_df[usd_df.index >= two_years_ago]
trends_df = trends_df[trends_df.index >= two_years_ago]

# یافتن تاریخ‌های ناقص
missing_dates = usd_df.index.difference(trends_df.index)
# پر کردن ناقص‌ها
if not missing_dates.empty:
    new_trends = fetch_missing_trends(missing_dates)
    trends_df = pd.concat([trends_df, new_trends]).sort_index()
    trends_df = trends_df.reindex(usd_df.index).ffill().bfill()

# ادغام داده‌ها
combined_df = pd.merge(
    usd_df, trends_df, left_index=True, right_index=True, how='inner'
)
combined_df = combined_df.ffill().bfill()

# سری قیمت دلار (برای مدل)
price_series = combined_df['price']

# مدل ARIMA و پیش‌بینی دو روز آینده
model = auto_arima(price_series, seasonal=False, suppress_warnings=True)
forecast = model.predict(n_periods=2)
forecast_dates = [price_series.index[-1] + timedelta(days=i) for i in range(1, 3)]
forecast_vals = list(forecast)

# نمایش نمودار
st.subheader("📊 Historical Data & 2-Day Forecast")
fig, ax = plt.subplots(figsize=(12, 6))
# داده‌های تاریخی
ax.plot(price_series.index, price_series.values, label='Historical Price', color='blue')
# خط جداکننده
ax.axvline(price_series.index[-1], color='gray', linestyle='--')
# نقاط پیش‌بینی
for i, (d, v) in enumerate(zip(forecast_dates, forecast_vals), start=1):
    ax.scatter(d, v, color='red')
    ax.annotate(
        f'Day+{i}: {v:,.0f}',
        xy=(d, v), xytext=(0, 10), textcoords='offset points',
        ha='center', fontsize=9,
        arrowprops=dict(arrowstyle='->', color='red')
    )
ax.set_title('USD Free Market Rate Forecast')
ax.legend(['Historical', 'Threshold'] + [f'Forecast Day+{i}' for i in (1,2)])
ax.grid(True)
st.pyplot(fig)

# نمایش نتایج عددی
st.success(
    f"🔮 Forecast for {forecast_dates[0].date()}: {forecast_vals[0]:,.0f}"
)
st.success(
    f"🔮 Forecast for {forecast_dates[1].date()}: {forecast_vals[1]:,.0f}"
)

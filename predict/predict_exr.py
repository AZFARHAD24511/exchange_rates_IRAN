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
import os

# پیکربندی صفحه
st.set_page_config(page_title="پیش‌بینی نرخ دلار آزاد", layout="centered")
st.title("📈 پیش‌بینی نرخ دلار آزاد با داده‌های گوگل ترندز")

# آدرس فایل ترندز در گیتهاب
GITHUB_TRENDS_CSV_URL = 'https://raw.githubusercontent.com/AZFARHAD24511/exchange_rates_IRAN/main/predict/google_trends_daily.csv'
KEYWORDS = ['خرید دلار', 'فروش دلار']

# بارگذاری داده‌های دلار آزاد
@st.cache_data
def load_usd_data():
    ts = int(datetime.now().timestamp() * 1000)
    url = f"https://api.tgju.org/v1/market/indicator/summary-table-data/price_dollar_rl?period=all&mode=full&ts={ts}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers)
    data = r.json()['data']
    records = []
    for row in data:
        try:
            price = float(row[0].replace(',', '').replace('<span class="high" dir="ltr">', '').replace('</span>', ''))
            date = datetime.strptime(row[6], "%Y/%m/%d")
            records.append({'date': date, 'price': price})
        except:
            continue
    df = pd.DataFrame(records).dropna()
    df = df.set_index('date').sort_index()
    return df

# بارگذاری ترندز از گیتهاب
@st.cache_data
def load_trends_csv():
    r = requests.get(GITHUB_TRENDS_CSV_URL)
    return pd.read_csv(StringIO(r.text), parse_dates=['date']).set_index('date')

# گرفتن داده‌های ناقص از گوگل ترندز
def fetch_missing_trends(missing_dates, geo='IR'):
    pytrends = TrendReq(hl='fa', tz=330)
    df_all = []
    for keyword in KEYWORDS:
        pytrends.build_payload([keyword], timeframe=f"{missing_dates.min().strftime('%Y-%m-%d')} {missing_dates.max().strftime('%Y-%m-%d')}", geo=geo)
        trend_data = pytrends.interest_over_time()
        if not trend_data.empty:
            df_all.append(trend_data[keyword])
    if df_all:
        return pd.concat(df_all, axis=1).loc[missing_dates]
    else:
        return pd.DataFrame(index=missing_dates)

# اجرای برنامه
with st.spinner("در حال بارگذاری داده‌ها..."):
    usd_df = load_usd_data()
    trends_df = load_trends_csv()

# فیلتر 2 سال آخر
two_years_ago = datetime.now() - timedelta(days=730)
usd_df = usd_df[usd_df.index >= two_years_ago]
trends_df = trends_df[trends_df.index >= two_years_ago]

# بررسی تاریخ‌های ناقص
missing_dates = usd_df.index.difference(trends_df.index)

# گرفتن داده‌های جدید از ترندز در صورت نیاز
if not missing_dates.empty:
    new_trends = fetch_missing_trends(missing_dates)
    trends_df = pd.concat([trends_df, new_trends]).sort_index()

# ادغام داده‌ها
combined_df = pd.merge(usd_df, trends_df, left_index=True, right_index=True, how='inner')
combined_df = combined_df.ffill().bfill()

# مدل ARIMA فقط روی قیمت دلار
model_data = combined_df['price']

# آموزش مدل و پیش‌بینی یک روز آینده
model = auto_arima(model_data, seasonal=False, suppress_warnings=True)
forecast = model.predict(n_periods=1)
forecast_date = model_data.index[-1] + timedelta(days=1)
forecast_value = forecast[0]

# نمایش نتایج
st.subheader("📊 نمودار نرخ دلار آزاد و پیش‌بینی یک روز آینده")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(model_data.index, model_data, label="داده‌های واقعی")
ax.axvline(model_data.index[-1], color='gray', linestyle='--')
ax.scatter(forecast_date, forecast_value, color='red', label="پیش‌بینی")
ax.annotate(f'{forecast_value:,.0f} تومان\n{forecast_date.date()}', 
            xy=(forecast_date, forecast_value),
            xytext=(-60, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, bbox=dict(boxstyle='round', fc='yellow', alpha=0.5))
ax.set_title("پیش‌بینی نرخ دلار آزاد")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# نمایش مقدار پیش‌بینی
st.success(f"🔮 پیش‌بینی نرخ دلار آزاد برای {forecast_date.date()} برابر است با: **{forecast_value:,.0f} تومان**")

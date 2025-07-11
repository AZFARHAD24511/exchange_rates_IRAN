import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
import arabic_reshaper
from zoneinfo import ZoneInfo
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pytrends.request import TrendReq
import math

# تنظیم صفحه Streamlit
st.set_page_config(page_title="نرخ دلار آزاد و نیما و پیش‌بینی فردا", layout="wide")
st.markdown("""
---
© 2025 Dr. Farhadi. All rights reserved.  
This application was developed by **Dr. Farhadi**, Ph.D. in *Economics (Econometrics)* and *Data Science*.  
All trademarks and intellectual property are protected. ™
""")

# تابع نمایش صحیح متن فارسی برای matplotlib
def _(text):
    return get_display(arabic_reshaper.reshape(text))

# تعریف دوره‌های ریاست جمهوری
presidents = [
    {'name': 'روحانی', 'start': '2013-08-04', 'end': '2021-08-04', 'color': 'purple'},
    {'name': 'رئیسی', 'start': '2021-08-05', 'end': '2025-08-05', 'color': 'green'},
    {'name': 'پزشکیان', 'start': '2025-08-06', 'end': '2033-08-06', 'color': 'turquoise'}
]
for p in presidents:
    p['start'] = datetime.strptime(p['start'], '%Y-%m-%d')
    p['end'] = datetime.strptime(p['end'], '%Y-%m-%d')

# کلاس اسکرپ قیمت ارز
class CurrencyScraper:
    currencies_dict = {
        'USD_Azad': 'price_dollar_rl',
        'USD_Nima': 'nima_buy_usd'
    }
    api_base_url = 'https://api.tgju.org/v1/market/indicator/summary-table-data/'
    @staticmethod
    def clean_price(price_str: str) -> float:
        return float(
            price_str.replace('<span class="high" dir="ltr">', '')
                     .replace('<span class="low" dir="ltr">', '')
                     .replace('</span>', '')
                     .replace(',', '')
        )
    @staticmethod
    def scrape(currency: str) -> pd.DataFrame:
        code = CurrencyScraper.currencies_dict[currency]
        ts = int(datetime.now().timestamp() * 1000)
        url = f"{CurrencyScraper.api_base_url}{code}?period=all&mode=full&ts={ts}"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = r.json().get('data', [])
        records = []
        for row in data:
            try:
                price = CurrencyScraper.clean_price(row[0])
                date = datetime.strptime(row[6], '%Y/%m/%d')
                records.append({'date': date, 'price': price})
            except:
                continue
        df = pd.DataFrame(records).set_index('date').sort_index()
        return df

# annotation
def annotate_last(ax, x, y, label, color='black'):
    ax.annotate(f'{label}: {y:,.0f}', xy=(x, y), xytext=(10, 10),
                textcoords='offset points', arrowprops=dict(arrowstyle='->', color=color),
                bbox=dict(boxstyle='round', fc='yellow', alpha=0.5))

# تابع دریافت Google Trends
@st.cache_data
def fetch_trends(term: str, start: datetime, end: datetime, geo: str = 'IR') -> pd.DataFrame:
    py = TrendReq(hl='fa', tz=+210)
    timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
    py.build_payload([term], timeframe=timeframe, geo=geo)
    df = py.interest_over_time()
    if 'isPartial' in df.columns:
        df = df.drop(columns=['isPartial'])
    df.index = df.index.tz_localize(None)
    return df.rename(columns={term: 'trend'})

# شروع اپ
st.title("مقایسه نرخ دلار آزاد و نیما و پیش‌بینی فردا")
with st.spinner("در حال دریافت داده‌ها..."):
    scraper = CurrencyScraper()
    df_azad = scraper.scrape('USD_Azad')
    df_nima = scraper.scrape('USD_Nima')
combined = pd.merge(df_azad['price'], df_nima['price'], left_index=True, right_index=True,
                    suffixes=('_azad','_nima'), how='inner')

if combined.empty:
    st.warning("داده‌ای برای تحلیل وجود ندارد")
else:
    # نمایش داده‌ها و دانلود
    combined['price_gap'] = combined['price_azad'] - combined['price_nima']
    combined['price_gap_ratio'] = combined['price_gap'] / combined['price_azad'] * 100
    st.download_button("📥 دانلود CSV", combined.to_csv().encode('utf-8'), "usd_data.csv", "text/csv")

    # نمایش چارت‌ها
    for title, series_list, colors in [
        (_('نرخ دلار آزاد و نیما'), ['price_azad','price_nima'], ['blue','orange']),
        (_('اختلاف نرخ'), ['price_gap'], ['black']),
        (_('نسبت اختلاف (%)'), ['price_gap_ratio'], ['green'])
    ]:
        st.subheader(title)
        fig, ax = plt.subplots(figsize=(12,4))
        for col, col_color in zip(series_list, colors):
            ax.plot(combined.index, combined[col], label=_(col), color=col_color)
        for p in presidents:
            if combined.index.min() <= p['start'] <= combined.index.max():
                ax.axvline(p['start'], color='gray', linestyle='--')
                ax.text(p['start'], ax.get_ylim()[1], _(p['name']), rotation=90,
                         verticalalignment='top', color=p['color'])
        annotate_last(ax, combined.index[-1], combined[series_list[0]].iloc[-1], _(series_list[0]), colors[0])
        ax.legend(); ax.grid(True)
        st.pyplot(fig)

    # پیش‌بینی فردا با Google Trends
    start, end = combined.index.min(), combined.index.max() + timedelta(days=1)
    trends = fetch_trends('دلار فردایی', start, end)
    data = combined.join(trends, how='inner').dropna()
    if not data.empty:
        train_y = data['price_azad'].iloc[:-1]
        train_x = data[['trend']].iloc[:-1]
        test_x = data[['trend']].iloc[-1:]
        model = SARIMAX(train_y, exog=train_x, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=1, exog=test_x)
        y_pred = pred.predicted_mean.iloc[0]
        y_true = data['price_azad'].iloc[-1]
        rmse = math.sqrt(mean_squared_error([y_true],[y_pred]))
        st.subheader(_('پیش‌بینی نرخ دلار فردا'))
        st.write(f"{_('مقدار واقعی')}: {y_true:,.0f}")
        st.write(f"{_('مقدار پیش‌بینی‌شده')}: {y_pred:,.0f}")
        st.write(f"{_('RMSE')} : {rmse:.2f}")
        fig_f, ax_f = plt.subplots(figsize=(10,4))
        ax_f.plot(data.index[-30:], data['price_azad'][-30:], label=_('واقعی'))
        ax_f.scatter(data.index[-1], y_pred, color='red', label=_('پیش‌بینی'))
        ax_f.set_title(_('نمودار پیش‌بینی'))
        ax_f.legend(); ax_f.grid(True)
        st.pyplot(fig_f)
    else:
        st.info(_('داده کافی برای پیش‌بینی موجود نیست.'))

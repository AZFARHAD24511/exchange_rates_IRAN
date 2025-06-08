import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
import arabic_reshaper
from zoneinfo import ZoneInfo
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# تنظیم صفحه
st.set_page_config(page_title="نرخ دلار آزاد و نیما", layout="wide")

# تابع نمایش متن فارسی
def _(text):
    return get_display(arabic_reshaper.reshape(text))

# کلاس اسکرپ نرخ ارز
class CurrencyScraper:
    currencies_dict = {
        'USD_Azad': 'price_dollar_rl',
        'USD_Nima': 'nima_buy_usd'
    }
    api_base_url = 'https://api.tgju.org/v1/market/indicator/summary-table-data/'

    @staticmethod
    def clean_price(price_str: str) -> float:
        return float(price_str.replace('<span class="high" dir="ltr">', '').replace('<span class="low" dir="ltr">', '').replace('</span>', '').replace(',', ''))

    @staticmethod
    def scrape_currency(currency: str) -> pd.DataFrame:
        code = CurrencyScraper.currencies_dict[currency]
        ts = int(datetime.now().timestamp() * 1000)
        url = f"{CurrencyScraper.api_base_url}{code}?period=all&mode=full&ts={ts}"
        headers = {'User-Agent': 'Mozilla/5.0'}

        r = requests.get(url, headers=headers)
        data = r.json()

        records = []
        for row in data['data']:
            try:
                price = CurrencyScraper.clean_price(row[0])
                date = datetime.strptime(row[6], '%Y/%m/%d')
                records.append({'date': date, 'price': price})
            except:
                continue

        df = pd.DataFrame(records)
        df = df.set_index('date').sort_index()
        return df

    @staticmethod
    def test_stationarity(series):
        return adfuller(series)[1]

    @staticmethod
    def johansen_test(df):
        result = coint_johansen(df, det_order=0, k_ar_diff=1)
        return result.lr1, result.cvt

# تابع نمایش آخرین مقدار
def annotate_last(ax, x, y, label, color='black'):
    ax.annotate(f'{label}: {y:,.0f}', xy=(x, y), xytext=(10, 10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=color),
                bbox=dict(boxstyle='round', fc='yellow', alpha=0.5))

# شروع برنامه Streamlit
st.title(_("مقایسه نرخ دلار آزاد و نیما"))

try:
    with st.spinner(_("در حال دریافت داده‌ها...")):
        scraper = CurrencyScraper()
        df_azad = scraper.scrape_currency('USD_Azad')
        df_nima = scraper.scrape_currency('USD_Nima')

    combined_df = pd.merge(
        df_azad['price'], df_nima['price'],
        left_index=True, right_index=True,
        suffixes=('_azad', '_nima'),
        how='inner'
    )

    if combined_df.empty:
        st.warning(_("داده‌ای برای تحلیل وجود ندارد"))
    else:
        # ذخیره فایل CSV
        today_str = datetime.now().strftime('%Y%m%d')
        os.makedirs('data', exist_ok=True)
        combined_df.to_csv(f'data/usd_data_{today_str}.csv')

        # محاسبه اختلاف و نسبت
        combined_df['price_gap'] = combined_df['price_azad'] - combined_df['price_nima']
        combined_df['price_gap_ratio'] = (combined_df['price_gap'] / combined_df['price_azad']) * 100

        # تست‌های آماری
        p_azad = scraper.test_stationarity(combined_df['price_azad'])
        p_nima = scraper.test_stationarity(combined_df['price_nima'])
        johansen_stat, johansen_crit = scraper.johansen_test(combined_df[['price_azad', 'price_nima']])

        # نمایش آمار تست‌ها
        st.subheader(_("نتایج آزمون ایستایی"))
        st.write(_(f"ADF دلار آزاد: p-value = {p_azad:.4f}"))
        st.write(_(f"ADF دلار نیما: p-value = {p_nima:.4f}"))

        st.subheader(_("آزمون هم‌جمعی یوهانسن"))
        st.write(_(f"آمار آزمون: {johansen_stat}"))
        st.write(_(f"مقادیر بحرانی:\n{johansen_crit}"))

        # نمودار اول: قیمت دلار
        st.subheader(_("نمودار نرخ دلار آزاد و نیما"))

        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(combined_df.index, combined_df['price_azad'], label=_('دلار آزاد'), color='blue')
        ax1.plot(combined_df.index, combined_df['price_nima'], label=_('دلار نیما'), color='orange')
        annotate_last(ax1, combined_df.index[-1], combined_df['price_azad'].iloc[-1], _('دلار آزاد'), 'blue')
        annotate_last(ax1, combined_df.index[-1], combined_df['price_nima'].iloc[-1], _('دلار نیما'), 'orange')
        ax1.set_title(_('مقایسه نرخ دلار آزاد و نیما'))
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        # نمودار دوم: اختلاف قیمت
        st.subheader(_("اختلاف نرخ دلار آزاد و نیما"))

        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(combined_df.index, combined_df['price_gap'], label=_('اختلاف قیمت'), color='black')
        annotate_last(ax2, combined_df.index[-1], combined_df['price_gap'].iloc[-1], _('اختلاف قیمت'), 'red')
        ax2.set_title(_('اختلاف بین دلار آزاد و دلار نیما'))
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # نمودار سوم: نسبت اختلاف به درصد
        st.subheader(_("نسبت اختلاف به دلار آزاد (درصد)"))

        fig3, ax3 = plt.subplots(figsize=(14, 6))
        ax3.plot(combined_df.index, combined_df['price_gap_ratio'], label=_('نسبت اختلاف'), color='green')
        annotate_last(ax3, combined_df.index[-1], combined_df['price_gap_ratio'].iloc[-1], _('درصد اختلاف'), 'green')
        ax3.set_title(_('نسبت اختلاف دلار آزاد و نیما به درصد'))
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)

except Exception as e:
    st.error(f"خطا در اجرای برنامه: {e}")

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

# تنظیم صفحه Streamlit
st.set_page_config(page_title="نرخ دلار آزاد و نیما", layout="wide")

# تابع نمایش صحیح متن فارسی برای matplotlib
def _(text):
    return get_display(arabic_reshaper.reshape(text))

# تعریف دوره‌های ریاست جمهوری به عنوان لیست از دیکشنری‌ها
# در این مثال، دوره‌های نمونه ذکر شده‌اند؛ شما می‌توانید آن‌ها را تغییر دهید.
presidents = [
    {'name': 'روحانی', 'start': '2013-08-04', 'end': '2021-08-04', 'color': 'purple'},
    {'name': 'رئیسی', 'start': '2021-08-05', 'end': '2025-08-05', 'color': 'green'},
    {'name': 'پزشکیان', 'start': '2025-08-06', 'end': '2033-08-06', 'color': 'turquoise'}  # تاریخ فرضی
]
# تبدیل تاریخ‌های هر دوره به datetime
for p in presidents:
    p['start'] = datetime.strptime(p['start'], '%Y-%m-%d')
    p['end'] = datetime.strptime(p['end'], '%Y-%m-%d')

# کلاس اسکرپ نرخ ارز
class CurrencyScraper:
    currencies_dict = {
        'USD_Azad': 'price_dollar_rl',
        'USD_Nima': 'nima_buy_usd'
    }
    api_base_url = 'https://api.tgju.org/v1/market/indicator/summary-table-data/'

    @staticmethod
    def clean_price(price_str: str) -> float:
        return float(price_str.replace('<span class="high" dir="ltr">', '')
                         .replace('<span class="low" dir="ltr">', '')
                         .replace('</span>', '').replace(',', ''))

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
            except Exception as e:
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

# تابع افزودن حاشیه‌نگاری (Annotation) آخرین مقدار بر روی نمودار
def annotate_last(ax, x, y, label, color='black'):
    ax.annotate(f'{label}: {y:,.0f}', xy=(x, y), xytext=(10, 10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=color),
                bbox=dict(boxstyle='round', fc='yellow', alpha=0.5))

# شروع برنامه Streamlit
st.title("مقایسه نرخ دلار آزاد و نیما")

try:
    with st.spinner("در حال دریافت داده‌ها..."):
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
        st.warning("داده‌ای برای تحلیل وجود ندارد")
    else:
        # ذخیره فایل CSV
        today_str = datetime.now().strftime('%Y%m%d')
        os.makedirs('data', exist_ok=True)
        combined_df.to_csv(f'data/usd_data_{today_str}.csv')
        csv_data = combined_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="📥 دانلود داده‌های پردازش‌شده (CSV)",
            data=csv_data,
            file_name=f'usd_data_{today_str}.csv',
            mime='text/csv'
        )

        # محاسبه اختلاف قیمت و نسبت اختلاف
        combined_df['price_gap'] = combined_df['price_azad'] - combined_df['price_nima']
        combined_df['price_gap_ratio'] = (combined_df['price_gap'] / combined_df['price_azad']) * 100

        # تست‌های آماری
        p_azad = scraper.test_stationarity(combined_df['price_azad'])
        p_nima = scraper.test_stationarity(combined_df['price_nima'])
        johansen_stat, johansen_crit = scraper.johansen_test(combined_df[['price_azad', 'price_nima']])

        # نمایش آمار در Streamlit (متون خام، بدون تابع _())
        st.subheader("نتایج آزمون ایستایی")
        st.write(f"ADF دلار آزاد : p-value = {p_azad:.4f}")
        st.write(f"ADF دلار نیما: p-value = {p_nima:.4f}")

        st.subheader("آزمون هم‌جمعی یوهانسن")
        st.write(f"آمار آزمون: {johansen_stat}")
        st.write(f"مقادیر بحرانی:\n{johansen_crit}")

        # نمودار ۱: نمودار قیمت‌ها
        st.subheader("نمودار نرخ دلار آزاد و نیما")
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(combined_df.index, combined_df['price_azad'], label=_('دلار آزاد'), color='blue')
        ax1.plot(combined_df.index, combined_df['price_nima'], label=_('دلار نیما'), color='orange')

        # اضافه کردن خطوط عمودی و برچسب دوره‌های ریاست جمهوری
        for p in presidents:
            # اگر تاریخ شروع دوره در بازه داده‌ها قرار داشت
            if combined_df.index.min() <= p['start'] <= combined_df.index.max():
                ax1.axvline(p['start'], color='gray', linestyle='--', linewidth=1)
                # قرار دادن نام دوره در بالای خط (چرخش 90 درجه)
                ax1.text(p['start'], ax1.get_ylim()[1], _(p['name']), rotation=90,
                         verticalalignment='top', fontsize=8, color=p['color'])
        # افزودن حاشیه‌نگاری آخرین مقدار
        annotate_last(ax1, combined_df.index[-1], combined_df['price_azad'].iloc[-1], _('دلار آزاد'), 'blue')
        annotate_last(ax1, combined_df.index[-1], combined_df['price_nima'].iloc[-1], _('دلار نیما'), 'orange')

        ax1.set_title(_('مقایسه نرخ دلار آزاد و نیما'))
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        # نمودار ۲: نمودار اختلاف قیمت
        st.subheader("نمودار اختلاف نرخ دلار آزاد و نیما")
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(combined_df.index, combined_df['price_gap'], label=_('اختلاف قیمت'), color='black')

        # اضافه کردن خطوط دوره‌های ریاست جمهوری
        for p in presidents:
            if combined_df.index.min() <= p['start'] <= combined_df.index.max():
                ax2.axvline(p['start'], color='gray', linestyle='--', linewidth=1)
                ax2.text(p['start'], ax2.get_ylim()[1], _(p['name']), rotation=90,
                         verticalalignment='top', fontsize=8, color=p['color'])

        annotate_last(ax2, combined_df.index[-1], combined_df['price_gap'].iloc[-1], _('اختلاف قیمت'), 'red')
        ax2.set_title(_('اختلاف بین دلار آزاد و دلار نیما'))
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # نمودار ۳: نمودار نسبت اختلاف به درصد
        st.subheader("نمودار نسبت اختلاف به دلار آزاد (درصد)")
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        ax3.plot(combined_df.index, combined_df['price_gap_ratio'], label=_('نسبت اختلاف'), color='green')

        # اضافه کردن خطوط دوره‌های ریاست جمهوری
        for p in presidents:
            if combined_df.index.min() <= p['start'] <= combined_df.index.max():
                ax3.axvline(p['start'], color='gray', linestyle='--', linewidth=1)
                ax3.text(p['start'], ax3.get_ylim()[1], _(p['name']), rotation=90,
                         verticalalignment='top', fontsize=8, color=p['color'])

        annotate_last(ax3, combined_df.index[-1], combined_df['price_gap_ratio'].iloc[-1], _('درصد اختلاف'), 'green')
        ax3.set_title(_('نسبت اختلاف دلار آزاد و نیما به درصد'))
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)

except Exception as e:
    st.error(f"خطا در اجرای برنامه: {e}")

#!/usr/bin/env python3
# forecast_with_trends.py

import requests
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pytrends.request import TrendReq

# 1. تابع اسکرپ قیمت دلار از API
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

    def scrape(self, code_key: str) -> pd.DataFrame:
        code = self.currencies_dict[code_key]
        ts = int(datetime.now().timestamp() * 1000)
        url = f"{self.api_base_url}{code}?period=all&mode=full&ts={ts}"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = r.json()['data']
        records = []
        for row in data:
            try:
                price = self.clean_price(row[0])
                date = datetime.strptime(row[6], '%Y/%m/%d')
                records.append((date, price))
            except:
                continue
        df = pd.DataFrame(records, columns=['date','price']).set_index('date').sort_index()
        return df

# 2. تابع دریافت Google Trends
def fetch_trends(term: str, start: datetime, end: datetime, geo: str = 'IR') -> pd.DataFrame:
    py = TrendReq(hl='fa', tz=+210)
    timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
    py.build_payload([term], timeframe=timeframe, geo=geo)
    df = py.interest_over_time().drop(columns=['isPartial'])
    df.index = df.index.tz_localize(None)
    return df.rename(columns={term: 'trend'})

def main():
    # الف) اسکرپ دو سری دلار
    scraper = CurrencyScraper()
    df_azad = scraper.scrape('USD_Azad')
    df_nima = scraper.scrape('USD_Nima')

    # ب) ترکیب دو سری
    df = pd.merge(df_azad, df_nima, left_index=True, right_index=True,
                  suffixes=('_azad','_nima'), how='inner')

    # ج) بازه زمانی برای ترند
    start, end = df.index.min(), df.index.max()+timedelta(days=1)
    trends = fetch_trends('دلار فردایی', start, end)

    # د) ترکیب با ترند
    data = df.join(trends, how='inner').dropna()

    if data.empty:
        print("No overlapping data between rates and trends.")
        return

    # ه) تقسیم train/test
    series = data['price_azad']
    exog  = data[['trend']]
    train_s, test_s = series.iloc[:-1], series.iloc[-1:]
    train_x, test_x = exog.iloc[:-1], exog.iloc[-1:]

    # و) مدل SARIMAX
    model = SARIMAX(train_s, exog=train_x, order=(1,1,1),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    # ز) پیش‌بینی یک قدم جلو
    pred = res.get_forecast(steps=1, exog=test_x)
    y_pred = pred.predicted_mean.iloc[0]
    y_true = test_s.iloc[0]
    rmse = mean_squared_error([y_true],[y_pred], squared=False)

    # ح) چاپ نتایج
    print(f"True value (last day):    {y_true:,.2f}")
    print(f"Forecasted value (next):  {y_pred:,.2f}")
    print(f"RMSE:                     {rmse:.2f}")

    # ط) رسم نمودار
    recent = series[-30:]
    plt.figure(figsize=(10,5))
    plt.plot(recent.index, recent.values, label='Historical Price')
    plt.scatter(test_s.index, [y_pred], color='red', label='Forecast Tomorrow')
    plt.title('USD Azad Price & Tomorrow Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


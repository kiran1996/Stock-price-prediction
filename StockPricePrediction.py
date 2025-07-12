import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Set time range
end = datetime.today()
start = end - timedelta(days=90)

# Major Indian stocks on NSE
stock_symbols = {
    "INFY": "INFY.NS",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS"
}

def fetch_news_headlines(query, days=90):
    news_data = []
    for d in range(days):
        search_date = end - timedelta(days=d)
        url = f"https://news.google.com/rss/search?q={query}+when:{d}d&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.content, features="xml")
            items = soup.find_all('item')
            for item in items:
                news_data.append({
                    'Date': pd.to_datetime(search_date).normalize(),
                    'Title': item.title.text
                })
        except:
            continue
    df_news = pd.DataFrame(news_data)
    if not df_news.empty:
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        df_news['Date'] = df_news['Date'].dt.normalize()
    return df_news

def get_features_and_target(symbol_name, ticker):
    # 1. Fetch stock data
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df = df.dropna()
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()

    # 2. Fetch news + sentiment
    df_news = fetch_news_headlines(symbol_name)
    if df_news.empty:
        df['Sentiment'] = 0  # fallback
    else:
        df_news['Sentiment'] = df_news['Title'].apply(lambda x: sid.polarity_scores(x)['compound'])
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        df_news['Date'] = df_news['Date'].dt.normalize()
        sentiment = df_news.groupby('Date')['Sentiment'].mean().reset_index()
        df = pd.merge(df, sentiment, on='Date', how='left')
        df['Sentiment'].fillna(0, inplace=True)

    # 3. Create lag features for time series prediction
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_MA20'] = df['MA20'].shift(1)
    df['Prev_MA50'] = df['MA50'].shift(1)
    df['Prev_Sentiment'] = df['Sentiment'].shift(1)

    # 4. Target is next-day price
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    return df

# Combine data for all stocks
all_data = []
for symbol, ticker in stock_symbols.items():
    print(f"Processing {symbol}...")
    df = get_features_and_target(symbol, ticker)
    df['Stock'] = symbol
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# Model training: Predict next-day close price
features = ['Prev_Close', 'Prev_MA20', 'Prev_MA50', 'Prev_Sentiment']
X = df_all[features]
y = df_all['Target']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Evaluate model
df_all['Predicted'] = model.predict(X)
mae = mean_absolute_error(y, df_all['Predicted'])
print(f"\n📉 Mean Absolute Error: {mae:.2f} INR")

# Plot sample prediction for 1 stock
sample_stock = 'INFY'
sample = df_all[df_all['Stock'] == sample_stock].copy()
plt.figure(figsize=(12, 6))
plt.plot(sample['Date'], sample['Target'], label='Actual')
plt.plot(sample['Date'], sample['Predicted'], label='Predicted')
plt.title(f"{sample_stock} - Next Day Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

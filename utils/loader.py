import json
import os
from datetime import timedelta, datetime
import traceback
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pandas as pd
import requests
from sklearn.calibration import LabelEncoder
import utils.loader as loader
import utils.stocks as stocks
from sklearn.preprocessing import StandardScaler
import ast
import torch
from torch.utils.data import TensorDataset, DataLoader


load_dotenv()
# Data source parameters
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co"

# Time span (days) for each news API call
NEWS_TIME_INTERVAL = 10
# Maximum number of news items per API call
NEWS_LIMIT = 1000

# Data time range
START_DATE = datetime(2022, 1, 3)
END_DATE = datetime(2025, 5, 26)
START_DATE_90 = datetime(2022, 5, 16)


# Query remote API with given parameters
def query_data_api(params):
    params.update({"apikey": ALPHA_VANTAGE_API_KEY})
    endpoint = f"{ALPHA_VANTAGE_BASE_URL}/query"

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ API request error: {e}")
        return {"error": str(e)}


# Fetch news data for a given ticker and time range
def fetch_news_data(ticker, start, end):
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": start.strftime("%Y%m%dT%H%M"),
        "time_to": end.strftime("%Y%m%dT%H%M"),
        "sort": "EARLIEST",
        "limit": NEWS_LIMIT,
    }

    return query_data_api(params)


# Fetch adjusted stock price data
def fetch_price_data(ticker, latest=False):
    params = {
        "symbol": ticker,
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "outputsize": "compact" if latest else "full",
        "datatype": "json",
    }

    api_response = query_data_api(params)

    # Extract the "Time Series (Daily)" field from response
    return api_response.get("Time Series (Daily)", {})


# Store news data locally as CSV
def store_news_data(ticker, start_date, end_date):
    save_dir = os.path.join(os.getcwd(), "data", "news")
    os.makedirs(save_dir, exist_ok=True)

    # DataFrame to store all results
    all_news_df = pd.DataFrame(
        columns=["title", "time_published", "summary", "ticker_sentiment"]
    )

    print(f"📡 Downloading news data for [{ticker}]...")

    # Iterate through the time interval in chunks
    current_date = start_date

    while current_date < end_date:
        next_date = current_date + timedelta(days=NEWS_TIME_INTERVAL)
        api_response = fetch_news_data(ticker, current_date, min(next_date, END_DATE))

        current_date = next_date

        if "feed" not in api_response:
            continue

        news_data = api_response["feed"]
        df = pd.DataFrame(news_data)

        if df.empty:
            continue

        df = df[["title", "time_published", "summary", "ticker_sentiment"]]

        # Filter out sentiment records related to the target ticker only
        df["ticker_sentiment"] = df["ticker_sentiment"].apply(
            lambda sentiments: (
                [s for s in sentiments if s.get("ticker") == ticker]
                if isinstance(sentiments, list)
                else []
            )
        )
        df = df[df["ticker_sentiment"].str.len() > 0]

        if not df.empty:
            all_news_df = pd.concat([all_news_df, df], ignore_index=True)

    if all_news_df.empty:
        print(f"⚠️ Warning: No news data retrieved for [{ticker}]!")
        return

    csv_file_path = os.path.join(save_dir, f"{ticker}_news.csv")
    all_news_df.to_csv(csv_file_path, encoding="utf-8-sig", index=False)

    print(f"✅ News data saved to: {csv_file_path}")

    return all_news_df


# Store price data locally as CSV
def store_price_data(ticker):
    save_dir = os.path.join(os.getcwd(), "data", "price")
    os.makedirs(save_dir, exist_ok=True)

    print(f"📡 Downloading price data for [{ticker}]...")

    time_series = fetch_price_data(ticker)
    price_df = pd.DataFrame(time_series).transpose()

    price_df.rename_axis("date", inplace=True)
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()

    # Convert values to float
    price_df = price_df.astype(float)

    # Calculate adjustment factor for adjusted close price
    factor = price_df["5. adjusted close"] / price_df["4. close"]

    # Apply adjusted prices and overwrite original columns
    price_df["1. open"] *= factor
    price_df["2. high"] *= factor
    price_df["3. low"] *= factor
    price_df["4. close"] = price_df[
        "5. adjusted close"
    ]  # overwrite with adjusted close
    price_df["6. volume"] /= factor

    # Rename columns to standard format
    column_names = {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "6. volume": "volume",
    }
    price_df = price_df.rename(columns=column_names)
    price_df = price_df[list(column_names.values())]

    # Round to 2 decimal places for prices, integer for volume
    price_df[["open", "high", "low", "close"]] = price_df[
        ["open", "high", "low", "close"]
    ].round(2)
    price_df["volume"] = price_df["volume"].round().astype(int)

    # Save to CSV
    csv_file_path = os.path.join(save_dir, f"{ticker}_price.csv")
    price_df.to_csv(csv_file_path, encoding="utf-8-sig", index=True)

    print(f"✅ Price data saved to: {csv_file_path}")

    return price_df


# Load news data from local CSV, fetch if missing
def load_news_data(ticker):
    base_dir = os.path.join(os.getcwd(), "data")

    news_file = os.path.join(base_dir, "news", f"{ticker}_news.csv")
    if os.path.exists(news_file):
        news_df = pd.read_csv(news_file, encoding="utf-8-sig")
    else:
        news_df = store_news_data(ticker, START_DATE, END_DATE)

    return news_df


# Load price data from local CSV, fetch if missing
def load_price_data(ticker):
    base_dir = os.path.join(os.getcwd(), "data")

    price_file = os.path.join(base_dir, "price", f"{ticker}_price.csv")
    if os.path.exists(price_file):
        price_df = pd.read_csv(
            price_file, encoding="utf-8-sig", index_col="date", parse_dates=True
        )
    else:
        # price_df = store_price_data(ticker)
        price_df = None

    return price_df


# Store both price and news data for a given ticker
def store_and_update_data(ticker):
    try:
        store_price_data(ticker)
    except Exception as e:
        print(f"❌ Failed to fetch price data for [{ticker}]")
        traceback.print_exc()

    try:
        store_news_data(ticker, START_DATE, END_DATE)
    except Exception as e:
        print(f"❌ Failed to fetch news data for [{ticker}]")
        traceback.print_exc()


# Batch update for multiple tickers
def store_all_ticker(tickers):
    for ticker in tickers:
        store_and_update_data(ticker)


# Calculate technical indicators for a given ticker
def calculate_indicators(ticker):
    base_dir = os.path.join(os.getcwd(), "data")
    price_file = os.path.join(base_dir, "price", f"{ticker}_price.csv")

    # Skip if file does not exist
    if not os.path.exists(price_file):
        print(f"Skipping {ticker}: missing data file.")
        return None

    price_df = pd.read_csv(price_file, encoding="utf-8-sig")

    # Convert date column to datetime and sort
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df.set_index("date", inplace=True)
    price_df = price_df.sort_index()

    # Restrict data to the defined END_DATE
    price_df = price_df.loc[:END_DATE]

    # Moving averages
    price_df["ma_5"] = price_df["close"].rolling(window=5).mean()
    price_df["ma_20"] = price_df["close"].rolling(window=20).mean()
    price_df["ma_90"] = price_df["close"].rolling(window=90).mean()

    # True Range (TR)
    price_df["prev_close"] = price_df["close"].shift(1)
    price_df["tr"] = price_df[["high", "prev_close"]].max(axis=1) - price_df[
        ["low", "prev_close"]
    ].min(axis=1)

    # Average True Range (ATR, 14-day)
    price_df["atr_14"] = price_df["tr"].rolling(window=14).mean()

    # 20-day rolling std of close price
    price_df["close_std_20"] = price_df["close"].rolling(window=20).std()

    # RSI (14-day)
    delta = price_df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    price_df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = price_df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = price_df["close"].ewm(span=26, adjust=False).mean()
    price_df["macd"] = ema_12 - ema_26
    price_df["macd_signal"] = price_df["macd"].ewm(span=9, adjust=False).mean()
    price_df["macd_hist"] = price_df["macd"] - price_df["macd_signal"]

    # Bollinger Bands (20-day)
    ma_20 = price_df["close"].rolling(window=20).mean()
    std_20 = price_df["close"].rolling(window=20).std()
    price_df["bb_upper"] = ma_20 + (2 * std_20)
    price_df["bb_middle"] = ma_20
    price_df["bb_lower"] = ma_20 - (2 * std_20)

    # Price Rate of Change (ROC, 10-day)
    price_df["roc_10"] = price_df["close"].pct_change(periods=10)

    # Volume Moving Averages
    price_df["volume_sma_5"] = price_df["volume"].rolling(window=5).mean()
    price_df["volume_sma_20"] = price_df["volume"].rolling(window=20).mean()

    # Remove intermediate columns
    price_df.drop(columns=["prev_close"], inplace=True)

    # Round float columns to 2 decimals
    float_cols = price_df.select_dtypes(include=["float64", "float32"]).columns
    price_df[float_cols] = price_df[float_cols].round(2)

    # Save updated file with indicators
    output_file = os.path.join(base_dir, "price", f"{ticker}_price.csv")
    price_df.to_csv(output_file, encoding="utf-8-sig")
    print(f"✅ Indicators calculated and saved for {ticker}.")


# Batch calculation for multiple tickers
def calculate_indicators_all(tickers):
    for ticker in tickers:
        try:
            calculate_indicators(ticker)
        except Exception as e:
            print(f"❌ Error processing ticker: {ticker}")
            print("Exception traceback:")
            traceback.print_exc()


# Selected features for modeling
features = [
    "close",  # Closing price (basic price reference)
    "atr_14",  # 14-day Average True Range (short-term volatility measure)
    "macd_hist",  # MACD histogram (momentum change strength, redundant signals removed)
    "volume_sma_20",  # 20-day Simple Moving Average of volume (smoothed volume trend)
    "rsi_14",  # 14-day Relative Strength Index (momentum / overbought-oversold indicator)
    "ma_20",  # 20-day Moving Average (trend indicator)
    "bb_middle",  # Middle band of Bollinger Bands (trend baseline)
]


def weighted_sentiment(row):
    """
    Convert raw ticker sentiment info (stored as stringified list of dicts)
    into a weighted sentiment score. The weight is based on both
    the sentiment score and its relevance to the stock.

    Args:
        row (str): Serialized sentiment data (list of dicts).

    Returns:
        float: Weighted sentiment score.
    """
    try:
        sentiment_info = ast.literal_eval(row)  # Safely parse string to Python object
        if sentiment_info and isinstance(sentiment_info, list):
            score = float(sentiment_info[0].get("ticker_sentiment_score", 0))
            relevance = float(sentiment_info[0].get("relevance_score", 0))
            return score * relevance
    except:
        return 0.0


def construct_all_stocks(start="2022-03-01", end="2025-05-01"):
    """
    Build a full stock dataset that merges technical indicators with news sentiment.
    For each stock in DJIA_STOCK_LIST:
        - Load price and news data
        - Calculate daily sentiment
        - Align with price data
        - Normalize features
        - Encode stock identifiers

    Args:
        start (str): Start date of the dataset.
        end (str): End date of the dataset.

    Returns:
        full_df (pd.DataFrame): Combined dataset for all stocks.
        stock_encoder (LabelEncoder): Encoder mapping stock symbols to IDs.
        scaler (StandardScaler): Fitted scaler for normalizing features.
    """
    all_df = []
    for stock in stocks.DJIA_STOCK_LIST:
        # Load price and news data
        price_df = loader.load_price_data(stock)
        news_df = loader.load_news_data(stock)

        # Compute weighted sentiment per news item
        news_df["weighted_sentiment"] = news_df["ticker_sentiment"].apply(
            weighted_sentiment
        )
        # Convert publication time into datetime (YYYYMMDD format)
        news_df["time_published"] = pd.to_datetime(
            news_df["time_published"].str[:8], format="%Y%m%d"
        )
        # Aggregate daily sentiment
        daily_sentiment = news_df.groupby("time_published")["weighted_sentiment"].mean()
        daily_sentiment = daily_sentiment.loc[start:end]

        # Compute daily returns
        price_df["return"] = 100 * (price_df["close"] / price_df["close"].shift(1) - 1)
        price_df = price_df[features + ["return"]].dropna()
        price_df = price_df.loc[start:end]

        # Join with daily sentiment
        price_df = price_df.join(daily_sentiment.rename("daily_sentiment"), how="left")
        price_df["daily_sentiment"] = price_df["daily_sentiment"].fillna(0)

        # Add stock identifier column
        price_df["stock"] = stock
        all_df.append(price_df)

    # Merge all stock data
    full_df = pd.concat(all_df).reset_index()

    # Encode stock symbol → stock_id
    stock_encoder = LabelEncoder()
    full_df["stock_id"] = stock_encoder.fit_transform(full_df["stock"])

    # Normalize numerical features
    scaler = StandardScaler()
    feature_cols = features
    full_df[feature_cols] = scaler.fit_transform(full_df[feature_cols])

    return full_df, stock_encoder, scaler


def make_sequences(df, window=10):
    X, y, stock_id = [], [], []
    df = df.sort_values("date")
    for i in range(len(df) - window):
        x_seq = df.iloc[i : i + window][features + ["daily_sentiment"]].values
        y_label = df.iloc[i + window]["return"]
        X.append(x_seq)
        y.append(y_label)
        stock_id.append(df.iloc[i + window]["stock_id"])
    return X, y, stock_id


def prepare_dataloader(df, window=10, batch_size=64, shuffle=True):
    X, y, stock_id = make_sequences(df, window)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    stock_id = torch.tensor(stock_id, dtype=torch.long)
    dataset = TensorDataset(X, stock_id, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

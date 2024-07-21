import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import numpy as np
import json
import subprocess
import logging
import os
import yaml
import talib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from alpha_vantage.timeseries import TimeSeries  # 确保导入 TimeSeries
from alpha_vantage.techindicators import TechIndicators  # 确保导入 TechIndicators


# 加载配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

alpha_vantage_api_key = config.get('alpha_vantage_api_key')
fred_api_key = config.get('fred_api_key')

# 设置日志记录
log_dir = config.get('log_dir', './logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'fetch_news.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = config.get('data_dir', './data')
raw_data_dir = os.path.join(data_dir, 'raw')
os.makedirs(raw_data_dir, exist_ok=True)

# 新闻数据获取和处理
def fetch_yahoo_finance_news(symbol):
    try:
        url = f'https://finance.yahoo.com/quote/{symbol}/news?p={symbol}'
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [a.text for a in soup.find_all('h3', class_='Mb(5px)')]
        return headlines
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Yahoo Finance news: {e}")
        return []

def fetch_cnbc_news(symbol, count=10):
    url = f'https://www.cnbc.com/quotes/?symbol={symbol}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = soup.find_all('a', class_='headline')
    news = [headline.get_text() for headline in headlines[:count]]
    logging.info(f"Fetched {len(news)} CNBC news articles for {symbol}")
    return news

def analyze_news_sentiment(news):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(article)['compound'] for article in news]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    logging.info(f"Calculated average sentiment for news: {avg_sentiment}")
    return avg_sentiment

# 使用Yahoo Finance获取基本面数据
def fetch_yahoo_finance_fundamental_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        cashflow = ticker.cashflow

        # 初始化所有字段为 None
        capex = None
        operating_cashflow = None
        fcf = None
        p_fcf = None

        try:
            capex = cashflow.loc['Capital Expenditures'].sum()
            operating_cashflow = cashflow.loc['Total Cash From Operating Activities'].sum()
            fcf = operating_cashflow - capex
            current_price = info.get('currentPrice')
            market_cap = info.get('marketCap')
            if fcf is not None and fcf != 0:
                p_fcf = market_cap / fcf
            else:
                p_fcf = None
        except KeyError as e:
            logging.error(f"Error fetching specific Yahoo Finance data for {symbol}: {e}")
            print(f"Error fetching specific Yahoo Finance data for {symbol}: {e}")

        piotroski_f_score = calculate_piotroski_f_score(info)

        fundamental_data = {
            'CROCI': info.get('returnOnEquity', None),
            'EBITDA': info.get('ebitda', None),
            'EPS_Growth': info.get('earningsGrowth', None),
            'P/E': info.get('forwardPE', None),
            'P/FCF': p_fcf,
            'P/B': info.get('priceToBook', None),
            'P/S': info.get('priceToSalesTrailing12Months', None),
            'Gross_Margin': info.get('grossMargins', None),
            'Piotroski_F_Score': piotroski_f_score,
            'ROA': info.get('returnOnAssets', None),
            'Sales_QQ': info.get('revenueQuarterlyGrowth', None)
        }

        logging.info(f"Fundamental data for {symbol}: {fundamental_data}")
        print(f"Fundamental data for {symbol}: {fundamental_data}")

        return fundamental_data
    except Exception as e:
        logging.error(f"Error fetching Yahoo Finance data: {e}")
        print(f"Error fetching Yahoo Finance data: {e}")
        return {
            'CROCI': None,
            'EBITDA': None,
            'EPS_Growth': None,
            'P/E': None,
            'P/FCF': None,
            'P/B': None,
            'P/S': None,
            'Gross_Margin': None,
            'Piotroski_F_Score': None,
            'ROA': None,
            'Sales_QQ': None
        }

def calculate_piotroski_f_score(info):
    score = 0

    # 盈利能力
    net_income = info.get('netIncome')
    operating_cashflow = info.get('operatingCashflow')
    roa = info.get('returnOnAssets')
    if net_income > 0:
        score += 1
    if operating_cashflow > 0:
        score += 1
    if operating_cashflow > net_income:
        score += 1
    if roa > 0:
        score += 1

    # 杠杆、流动性和融资
    current_ratio = info.get('currentRatio')
    long_term_debt = info.get('longTermDebt')
    shares_outstanding = info.get('sharesOutstanding')
    if current_ratio > 1:
        score += 1
    if long_term_debt < info.get('longTermDebtYearAgo'):
        score += 1
    if shares_outstanding <= info.get('sharesOutstandingYearAgo'):
        score += 1

    # 运营效率
    gross_margin = info.get('grossMargins')
    asset_turnover = info.get('assetTurnover')
    if gross_margin > info.get('grossMarginsYearAgo'):
        score += 1
    if asset_turnover > info.get('assetTurnoverYearAgo'):
        score += 1

    return score

def calculate_slope(data):
    slopes = []
    for i in range(len(data)):
        if i == 0:
            slopes.append(0)
        else:
            slopes.append(data[i] - data[i - 1])
    return slopes

def fetch_alpha_vantage_technical_data(symbol):
    try:
        # 从 Yahoo Finance 获取历史股票数据
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='2y')  # 获取过去两年的数据
        logging.info(f"Fetched historical data for {symbol}")
        
        # 打印数据头部和尾部以验证数据
        logging.info(f"Historical data head: {data.head()}")
        logging.info(f"Historical data tail: {data.tail()}")
        # print(f"Historical data head: {data.head()}")
        # print(f"Historical data tail: {data.tail()}")
        
        # 检查数据的长度
        logging.info(f"Number of data points: {len(data)}")
        # print(f"Number of data points: {len(data)}")

        # 检查是否有足够的数据
        if len(data) < 252:
            logging.error(f"Not enough data to calculate 52-week high for {symbol}")
            # print(f"Not enough data to calculate 52-week high for {symbol}")
            return {}

        # 使用 ta-lib 计算技术指标
        close_prices = data['Close']
        logging.info(f"Close prices: {close_prices.describe()}")
        # print(f"Close prices: {close_prices.describe()}")

        rsi_data = talib.RSI(close_prices, timeperiod=14)
        logging.info(f"RSI data: {rsi_data}")
        # print(f"RSI data: {rsi_data}")
        macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        logging.info(f"MACD data: {macd}")
        # print(f"MACD data: {macd}")
        willr_data = talib.WILLR(data['High'], data['Low'], close_prices, timeperiod=14)
        logging.info(f"WILLR data: {willr_data}")
        # print(f"WILLR data: {willr_data}")
        mfi_data = talib.MFI(data['High'], data['Low'], close_prices, data['Volume'], timeperiod=14)
        logging.info(f"MFI data: {mfi_data}")
        # print(f"MFI data: {mfi_data}")

        high_52_week = close_prices.rolling(window=252).max()  # 52周高点
        logging.info(f"52-week high prices: {high_52_week.dropna().describe()}")
        # print(f"52-week high prices: {high_52_week.dropna().describe()}")

        current_price = close_prices.iloc[-1]
        logging.info(f"Current price: {current_price}")
        # print(f"Current price: {current_price}")

        if high_52_week.iloc[-1] == 0 or pd.isna(high_52_week.iloc[-1]):
            logging.error("52-week high price is zero or NaN, which is invalid")
            print("52-week high price is zero or NaN, which is invalid")
            return {}

        high_52_week_percent = (current_price / high_52_week.iloc[-1]) * 100
        logging.info(f"52-week high percent: {high_52_week_percent}")
        # print(f"52-week high percent: {high_52_week_percent}")

        technical_data = {
            '52_week_high_percent': high_52_week_percent,
            'rsi': rsi_data.tolist(),
            'rsi_slope': calculate_slope(rsi_data.tolist()),
            'macd': macd.tolist(),
            'macdsignal': macdsignal.tolist(),
            'macdhist': macdhist.tolist(),
            'willr': willr_data.tolist(),
            'mfi': mfi_data.tolist()
        }
        logging.info(f"Technical data for {symbol}: {technical_data}")
        # print(f"Technical data for {symbol}: {technical_data}")
        return technical_data
    except Exception as e:
        logging.error(f"Error fetching technical data: {e}")
        # print(f"Error fetching technical data: {e}")
        return {}
    
# 股票数据获取
def fetch_stock_data(symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        return data
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# 获取多个股票数据
def fetch_yfinance_data(tickers_list, start_date, end_date):
    try:
        data = yf.download(tickers_list, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        return data
    except Exception as e:
        logging.error(f"Error fetching multiple stock data: {e}")
        return pd.DataFrame()

# 获取社交媒体数据
def fetch_twitter_data(keyword, count=10):
    try:
        command = f"snscrape --jsonl --max-results {count} twitter-search '{keyword} since:2023-01-01'"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        tweets = [json.loads(line)['content'] for line in result.stdout.splitlines()]
        return tweets
    except Exception as e:
        logging.error(f"Error fetching Twitter data: {e}")
        return []

def fetch_reddit_data(subreddit, count=10):
    try:
        command = f"snscrape --jsonl --max-results {count} reddit-submission 'subreddit:{subreddit}'"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        posts = [json.loads(line)['title'] for line in result.stdout.splitlines()]
        return posts
    except Exception as e:
        logging.error(f"Error fetching Reddit data: {e}")
        return []

def fetch_facebook_data(page_id, access_token, count=10):
    try:
        url = f'https://graph.facebook.com/v10.0/{page_id}/posts?limit={count}&access_token={access_token}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        posts = [post['message'] for post in data['data'] if 'message' in post]
        return posts
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Facebook data: {e}")
        return []

def fetch_instagram_data(user_id, access_token, count=10):
    try:
        url = f'https://graph.instagram.com/{user_id}/media?fields=id,caption&limit={count}&access_token={access_token}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        posts = [post['caption'] for post in data['data'] if 'caption' in post]
        return posts
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Instagram data: {e}")
        return []

def fetch_linkedin_data(company_id, access_token, count=10):
    try:
        url = f'https://api.linkedin.com/v2/shares?q=owners&owners=urn:li:organization:{company_id}&count={count}&oauth2_access_token={access_token}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        posts = [post['text']['text'] for post in data['elements'] if 'text' in post]
        return posts
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching LinkedIn data: {e}")
        return []

# 获取 FRED 经济数据
def fetch_fred_data(series_id, api_key, start_date, end_date):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}"
    response = requests.get(url)
    data = response.json()['observations']
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['value'] = df['value'].astype(float)
    df.to_csv(f'data/raw/{series_id}_economic_data.csv')
    logging.info(f"Fetched economic data for series {series_id}")
    return df

# 综合数据获取
def fetch_all_data(symbol, tickers_list, start_date, end_date, fred_api_key, fb_page_id=None, ig_user_id=None, li_company_id=None, access_token=None):
    yahoo_news = fetch_yahoo_finance_news(symbol)
    cnbc_news = fetch_cnbc_news(symbol)
    twitter_news = fetch_twitter_data(symbol)
    reddit_news = fetch_reddit_data(symbol)
    facebook_posts = fetch_facebook_data(fb_page_id, access_token) if fb_page_id and access_token else []
    instagram_posts = fetch_instagram_data(ig_user_id, access_token) if ig_user_id and access_token else []
    linkedin_posts = fetch_linkedin_data(li_company_id, access_token) if li_company_id and access_token else []

    all_news = yahoo_news + cnbc_news + twitter_news + reddit_news + facebook_posts + instagram_posts + linkedin_posts
    sentiment_score = analyze_news_sentiment(all_news)

    stock_data = fetch_yfinance_data(tickers_list, start_date, end_date)
    fundamental_data = [fetch_yahoo_finance_fundamental_data(ticker) for ticker in tickers_list]
    technical_data = [fetch_alpha_vantage_technical_data(ticker) for ticker in tickers_list]
    economic_data = fetch_fred_data('GDP', fred_api_key, start_date, end_date)

    return {
        'news': all_news,
        'sentiment_score': sentiment_score,
        'stock_data': stock_data,
        'fundamental_data': fundamental_data,
        'technical_data': technical_data,
        'economic_data': economic_data
    }
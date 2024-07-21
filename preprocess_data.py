import pandas as pd
import numpy as np
import os
import logging
import json
import yaml
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 加载配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# 设置日志记录
log_dir = config['log_dir']
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'preprocess_data.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = config['data_dir']
raw_data_dir = os.path.join(data_dir, 'raw')
processed_data_dir = os.path.join(data_dir, 'processed')

# 获取情绪分析结果
def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    logging.info(f"Sentiment analysis successful for text: {text}")
    return sentiment_score

# 分析新闻数据
def analyze_news(news_file_path):
    with open(news_file_path, 'r') as f:
        news_data = f.readlines()

    sentiments = [sentiment_analysis(news) for news in news_data]

    sentiment_factor = sum(sentiments) / len(sentiments) if sentiments else 0
    return sentiment_factor

# 预处理股票数据并保存
def preprocess_and_save():
    stock_file_path = os.path.join(raw_data_dir, 'stock_data.csv')
    technical_file_path = os.path.join(raw_data_dir, 'technical_data.json')
    fundamental_file_path = os.path.join(raw_data_dir, 'fundamental_data.json')
    news_file_path = os.path.join(raw_data_dir, 'news_data.txt')
    economic_file_path = os.path.join(raw_data_dir, 'GDP_economic_data.csv')

    stock_data = pd.read_csv(stock_file_path)

    with open(technical_file_path, 'r') as f:
        technical_data_list = json.load(f)

    with open(fundamental_file_path, 'r') as f:
        fundamental_data_list = json.load(f)

    economic_data = pd.read_csv(economic_file_path, index_col='date', parse_dates=True)

    if 'Adj Close' in stock_data.columns:
        stock_data['Close'] = stock_data['Adj Close']

    sentiment_factor = analyze_news(news_file_path)
    stock_data['Sentiment'] = sentiment_factor

    logging.info(f"Technical data list: {technical_data_list}")

    # 确保 technical_data_list 中的每个元素是数据框
    for technical_data in technical_data_list:
        if '52_week_high_percent' in technical_data:
            stock_data['52_week_high_percent'] = technical_data['52_week_high_percent']
        technical_data_df = pd.DataFrame(technical_data)
        # 对齐索引
        technical_data_df = technical_data_df.reindex(stock_data.index)
        for col in technical_data_df.columns:
            if col != '52_week_high_percent':  # 避免重复覆盖
                stock_data[col] = technical_data_df[col].values
                logging.info(f"Processed {col} column: {technical_data_df[col].values}")

    logging.info(f"Stock data columns after processing technical data: {stock_data.columns}")

    # 处理 fundamental_data_list 中的每个元素
    for fundamental_data in fundamental_data_list:
        for key, value in fundamental_data.items():
            stock_data[key] = value if value is not None else 0  # 用0代替None
            logging.info(f"Processed {key} column: {value}")

    logging.info(f"Stock data columns after processing fundamental data: {stock_data.columns}")

    # 检查所有必需列是否存在
    required_columns = ['CROCI', 'EBITDA', 'EPS_Growth', 'P/E', 'P/FCF', 'P/B', 'P/S', 'Gross_Margin', 
                        'Piotroski_F_Score', 'ROA', 'Sales_QQ', '52_week_high_percent', 'rsi', 'rsi_slope', 
                        'macd', 'macdsignal', 'macdhist', 'willr', 'mfi', 'Sentiment']

    for col in required_columns:
        if col not in stock_data.columns:
            logging.error(f"Data must contain '{col}' column.")
            print(f"Data must contain '{col}' column.")
            raise ValueError(f"Data must contain '{col}' column.")

    # 确保 'Date' 列为 datetime 类型
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')

    stock_data = stock_data.merge(economic_data, how='left', left_on='Date', right_index=True)

    logging.info(f"Stock data columns after merging economic data: {stock_data.columns}")

    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
    stock_data = stock_data.select_dtypes(include=[np.number]).join(stock_data['Date'])

    processed_file_path = os.path.join(processed_data_dir, 'processed_stock_data.csv')
    stock_data.to_csv(processed_file_path, index=False)
    logging.info(f"Processed stock data saved to {processed_file_path}")

if __name__ == "__main__":
    preprocess_and_save()

import os
import sys
import logging
import yaml
import json

# 添加 scripts 目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from fetch_news import fetch_all_data, fetch_alpha_vantage_technical_data, fetch_yahoo_finance_fundamental_data
from preprocess_data import preprocess_and_save
from model import train_model
from generate_signals import main as generate_signals_main

# 加载配置文件
config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# 设置日志记录配置
log_dir = config.get('log_dir', './logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = config.get('data_dir', './data')
raw_data_dir = os.path.join(data_dir, 'raw')
os.makedirs(raw_data_dir, exist_ok=True)

# 主函数
if __name__ == "__main__":
    # 获取用户输入的股票符号
    stock_symbol = input("Enter the main stock symbol for news (e.g., AAPL): ")
    # 获取用户输入的股票符号列表
    tickers_list = input("Enter the list of stock symbols for financial data, separated by commas (e.g., AAPL,WMT,IBM): ").split(',')
    # 获取用户输入的起始日期
    start_date = input("Enter the start date for historical data (e.g., 2023-01-01): ")
    # 获取用户输入的结束日期
    end_date = input("Enter the end date for historical data (e.g., 2023-12-31): ")

    fb_page_id = input("Enter the Facebook page ID (or leave blank): ")
    ig_user_id = input("Enter the Instagram user ID (or leave blank): ")
    li_company_id = input("Enter the LinkedIn company ID (or leave blank): ")
    access_token = input("Enter the access token for Facebook, Instagram, and LinkedIn (or leave blank): ")

    fred_api_key = config['fred_api_key']

    # 第一步：抓取数据
    all_data = fetch_all_data(stock_symbol, tickers_list, start_date, end_date, fred_api_key, fb_page_id, ig_user_id, li_company_id, access_token)

    # 使用新的函数抓取基本面数据
    all_data['fundamental_data'] = [fetch_yahoo_finance_fundamental_data(ticker) for ticker in tickers_list]

    # 保存新闻数据
    news_file_path = os.path.join(raw_data_dir, 'news_data.txt')
    with open(news_file_path, 'w') as f:
        for news in all_data['news']:
            f.write(news + "\n")
    logging.info(f"News data saved to {news_file_path}")
    
    # 保存股票数据
    stock_file_path = os.path.join(raw_data_dir, 'stock_data.csv')
    all_data['stock_data'].to_csv(stock_file_path, index=False)
    logging.info(f"Stock data saved to {stock_file_path}")
    
    # 保存基本面数据
    fundamental_file_path = os.path.join(raw_data_dir, 'fundamental_data.json')
    with open(fundamental_file_path, 'w') as f:
        json.dump(all_data['fundamental_data'], f, indent=4)
    logging.info(f"Fundamental data saved to {fundamental_file_path}")
    print(f"Fundamental data saved to {fundamental_file_path}")

    # 打印 fundamental_data.json 的内容以进行检查
    with open(fundamental_file_path, 'r') as f:
        saved_fundamental_data = json.load(f)
    print("Saved fundamental data:", saved_fundamental_data)
    logging.info(f"Saved fundamental data: {saved_fundamental_data}")

    # 保存技术指标数据
    technical_data = [fetch_alpha_vantage_technical_data(ticker) for ticker in tickers_list]
    logging.info(f"Fetched technical data: {technical_data}")
    # print("Fetched technical data:", technical_data)

    technical_file_path = os.path.join(raw_data_dir, 'technical_data.json')
    with open(technical_file_path, 'w') as f:
        json.dump(technical_data, f)
    logging.info(f"Technical data saved to {technical_file_path}")

    # 保存经济数据
    economic_file_path = os.path.join(raw_data_dir, 'GDP_economic_data.csv')
    all_data['economic_data'].to_csv(economic_file_path)
    logging.info(f"Economic data saved to {economic_file_path}")

    # 第二步：预处理数据
    preprocess_and_save()

    # 第三步：训练模型
    train_model()

    # 第四步：生成交易信号并绘制结果
    generate_signals_main()

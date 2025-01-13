import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
from ib_insync import IB, Stock, util

# 历史数据获取函数
def fetch_historical_data(ticker, ib, config, save_data=True, period=None, interval=None):
    """
    获取历史数据并保存到本地文件。
    """
    try:
        period = period or config["data_fetch"]['period']
        interval = interval or config["data_fetch"]['interval']
        save_path = config["data_fetch"]["data_path"]["raw_data_path"]  # 确保 data_paths 可用

        contract = Stock(ticker, 'SMART', 'USD')
        ib.qualifyContracts(contract)

        print(f"[DEBUG] 开始获取历史数据 -> 股票代码: {ticker}，范围: {period}，间隔: {interval}...")
        
        # 使用 ib_insync 获取历史数据
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=period,
            barSizeSetting=interval,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        data = util.df(bars)
        
        if not data.empty:
            if save_data:
                os.makedirs(save_path, exist_ok=True)  # 确保路径存在
                file_name = os.path.join(save_path, f"{ticker}_{period}.parquet")
                data.to_parquet(file_name)
                print(f"[成功] 历史数据已保存 -> 文件路径: {file_name}")
            return data
        else:
            print(f"[错误] 股票代码无效或无历史数据 -> 股票代码: {ticker}")
            return pd.DataFrame()  # 返回一个空的 DataFrame 以供后续处理

    except Exception as e:
        print(f"[错误] 获取历史数据时出错 -> 股票代码: {ticker}, 错误信息: {e}")
        return pd.DataFrame()  # 返回空的 DataFrame

# 实时数据获取函数
def fetch_realtime_data(ticker, ib, config, retry_count=None):
    """
    获取实时数据并返回数据字典。
    如果在市场关闭时间，获取前一天的最后一个数据点。
    
    :param ticker: 股票代码
    :param ib: IB实例
    :param config: 配置字典
    :return: 包含实时数据的字典或None
    """
    retry_count = config["data_fetch"]['retry_count']

    for attempt in range(retry_count + 1):  # 包括0次尝试，所以循环次数是retry_count + 1
        try:
            now = datetime.now(pytz.timezone('America/New_York'))
            market_open = now.weekday() < 5 and 9.30 <= now.hour + now.minute/60 < 16  # 周一到周五，9:30 AM - 4:00 PM EST

            contract = Stock(ticker, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            
            if market_open:
                # 获取实时数据
                ticker_data = ib.reqMktData(contract)
                ib.sleep(1)  # 等待数据更新
                
                if ticker_data.marketPrice() is not None:
                    print(f"[DEBUG] 实时数据获取成功 -> 股票代码: {ticker}")
                    return {
                        "symbol": ticker,
                        "timestamp": datetime.now(),
                        "open": ticker_data.open,
                        "high": ticker_data.high,
                        "low": ticker_data.low,
                        "close": ticker_data.marketPrice(),
                        "volume": ticker_data.volume,
                    }
            else:
                # 市场关闭时，获取最后一个交易日的数据
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='1 D',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )
                data = util.df(bars)
                if not data.empty:
                    last_data = data.iloc[-1]
                    print(f"[DEBUG] 市场关闭，获取前一天数据成功 -> 股票代码: {ticker}")
                    return {
                        "symbol": ticker,
                        "timestamp": last_data.name,
                        "open": last_data["open"],
                        "high": last_data["high"],
                        "low": last_data["low"],
                        "close": last_data["close"],
                        "volume": last_data["volume"],
                    }
                else:
                    print(f"[DEBUG] 获取最近可用数据失败 -> 股票代码: {ticker}")
                    return None

            if attempt < retry_count:
                print(f"[信息] 获取数据失败，尝试重新获取 ({attempt+1}/{retry_count})")
            else:
                print(f"[错误] 获取数据失败，超过重试次数 -> 股票代码: {ticker}")
                return None
        except Exception as e:
            print(f"[错误] 获取数据时出错 -> 股票代码: {ticker}, 错误信息: {e}")
            if attempt < retry_count:
                print(f"[信息] 尝试重新获取数据 ({attempt+1}/{retry_count})")
            else:
                print(f"[错误] 获取数据失败，超过重试次数 -> 股票代码: {ticker}")
                return None

    return None


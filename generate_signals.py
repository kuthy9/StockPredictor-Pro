import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import logging
import yaml

# 加载配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# 设置日志记录
log_dir = config['log_dir']
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'generate_signals.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = config['data_dir']
processed_data_dir = os.path.join(data_dir, 'processed')

# 生成买卖信号函数
def generate_signals(data, window_short=30, window_long=60):
    signals = pd.DataFrame(index=np.arange(len(data)))
    signals['signal'] = 0.0
    signals['short_mavg'] = pd.Series(data[:, 0]).rolling(window=window_short, min_periods=1).mean()
    signals['long_mavg'] = pd.Series(data[:, 0]).rolling(window=window_long, min_periods=1).mean()
    signals['signal'][window_short:] = np.where(signals['short_mavg'][window_short:] > signals['long_mavg'][window_short:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    buy_signals = (signals[signals['positions'] == 1.0].index + 1).tolist()
    sell_signals = (signals[signals['positions'] == -1.0].index + 1).tolist()
    return buy_signals, sell_signals

def plot_results(data, trainPredict, testPredict, future_predictions, look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = data[['Close']].values.astype('float32')
    dataset = scaler.fit_transform(dataset)
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(data['Date']), scaler.inverse_transform(dataset)[:, 0], label='True Price')
    plt.plot(pd.to_datetime(data['Date'][look_back:look_back + len(trainPredict)]), trainPredict, label='Train Predict')
    plt.plot(pd.to_datetime(data['Date'][len(data) - len(testPredict) - look_back:len(data) - look_back]), testPredict, label='Test Predict')
    future_dates = pd.date_range(start=pd.to_datetime(data['Date'].iloc[-1]), periods=len(future_predictions) + 1)
    plt.plot(future_dates[1:], future_predictions, 'r--', label='Future Predict')
    plt.legend()
    plt.show()

def main():
    processed_file_path = os.path.join(processed_data_dir, 'processed_stock_data.csv')
    processed_data = pd.read_csv(processed_file_path)

    if 'Close' not in processed_data.columns:
        logging.error("Error: 'Close' column is missing in the processed data.")
        return

    buy_signals, sell_signals = generate_signals(processed_data[['Close']].values, window_short=30, window_long=60)

    logging.info(f"Buy signals at: {buy_signals}")
    logging.info(f"Sell signals at: {sell_signals}")

    processed_data['Buy_Signal'] = 0
    processed_data['Sell_Signal'] = 0
    processed_data.loc[buy_signals, 'Buy_Signal'] = 1
    processed_data.loc[sell_signals, 'Sell_Signal'] = 1

    processed_signals_file_path = os.path.join(processed_data_dir, 'processed_stock_data_with_signals.csv')
    processed_data.to_csv(processed_signals_file_path, index=False)
    logging.info(f"Processed stock data with signals saved to {processed_signals_file_path}")

    # 假设有未来预测数据
    future_predictions = np.random.rand(30)  # 替换为真实预测
    plot_results(processed_data, processed_data['Close'].values, processed_data['Close'].values, future_predictions, look_back=60)

if __name__ == "__main__":
    main()

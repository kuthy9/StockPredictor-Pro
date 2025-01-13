import pandas as pd
import numpy as np


def calculate_technical_indicators(df, indicators=None):
    """
    计算技术指标，支持动态选择。
    :param df: 包含股票数据的 DataFrame，必须包括 ['close', 'high', 'low', 'volume']
    :param indicators: 指定要计算的指标列表。如果为 None，计算所有指标。
    """
    df = df.copy() 

    # 检查数据完整性
    required_columns = ['close', 'high', 'low', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据缺少必要列: {col}，无法计算技术指标。")

    # 定义支持的技术指标
    all_indicators = {
        'sma_5': lambda df: calculate_sma5(df, column='close', window=5, fill_method='mean'),
        'sma_50': lambda df: calculate_sma50(df, column='close', window=5, fill_method='mean'),
        'ema_5': lambda df: df['close'].ewm(span=5, adjust=False).mean(),
        'rsi': lambda df: calculate_rsi(df, window=14),
        'macd': lambda df: calculate_macd(df),
        'bollinger': lambda df: calculate_bollinger_bands(df),
        'atr': lambda df: calculate_atr(df, window=14),
        'vwap': lambda df: calculate_vwap(df),
        'volume_oscillator': lambda df: calculate_volume_oscillator(df, short_window=5, long_window=20),
        'momentum': lambda df: df['close'] - df['close'].shift(10),
        'stochastic': lambda df: calculate_stochastic_oscillator(df, window=14),
        'cci': lambda df: calculate_cci(df, window=20),
        'roc': lambda df: calculate_roc(df, window=10),
        'williams_r': lambda df: calculate_williams_r(df, window=14),
        'adx': lambda df: calculate_adx(df, window=14)
    }

    # 选择要计算的指标
    indicators = indicators or list(all_indicators.keys())
    selected_indicators = {k: v for k, v in all_indicators.items() if k in indicators}

    # 开始计算
    print("开始计算技术指标...")
    for name, func in selected_indicators.items():
        try:
            print(f"计算 {name}...")
            result = func(df)
            if isinstance(result, pd.DataFrame):
                df = pd.concat([df, result], axis=1)
            else:
                df.loc[:, name] = result 
        except Exception as e:
            print(f"计算 {name} 失败: {e}")

    # 删除空值
    df.dropna(inplace=True)
    print("技术指标计算完成。")
    return df

def calculate_sma5(df, column='close', window=5, fill_method='mean'):
    sma = df[column].rolling(window=window).mean()

    # 填充 NaN 值
    if fill_method == 'mean':
        sma = sma.fillna(sma.mean())
    elif fill_method == 'median':
        sma = sma.fillna(sma.median())
    elif isinstance(fill_method, (int, float)):
        sma = sma.fillna(fill_method)
    else:
        raise ValueError("无效的填充方式，请选择 'mean', 'median' 或指定数值")

    return sma

def calculate_sma50(df, column='close', window=50, fill_method='mean'):
    sma = df[column].rolling(window=window).mean()

    # 填充 NaN 值
    if fill_method == 'mean':
        sma = sma.fillna(sma.mean())
    elif fill_method == 'median':
        sma = sma.fillna(sma.median())
    elif isinstance(fill_method, (int, float)):
        sma = sma.fillna(fill_method)
    else:
        raise ValueError("无效的填充方式，请选择 'mean', 'median' 或指定数值")

    return sma

def calculate_rsi(df, window=14):
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 使用均值填充 NaN 值（如果需要中位数，可以改为 rsi.fillna(rsi.median())）
    rsi = rsi.fillna(rsi.mean())
    return rsi


def calculate_macd(df):
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return pd.DataFrame({'macd_line': macd_line, 'signal_line': signal_line, 'macd_histogram': macd_histogram})


def calculate_bollinger_bands(df, window=20):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return pd.DataFrame({'upper_band': upper_band, 'lower_band': lower_band})


def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def calculate_vwap(df):
    return (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()


def calculate_volume_oscillator(df, short_window=5, long_window=20):
    short_ma = df['volume'].rolling(window=short_window).mean()
    long_ma = df['volume'].rolling(window=long_window).mean()
    return short_ma - long_ma


def calculate_stochastic_oscillator(df, window=14):
    lowest_low = df['low'].rolling(window=window).min()
    highest_high = df['high'].rolling(window=window).max()
    k_percent = (df['close'] - lowest_low) / (highest_high - lowest_low) * 100
    d_percent = k_percent.rolling(window=3).mean()
    return pd.DataFrame({'k%': k_percent, 'd%': d_percent})


def calculate_cci(df, window=20):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    return (df['close'] - rolling_mean) / (0.015 * rolling_std)


def calculate_roc(df, window=10):
    return ((df['close'] - df['close'].shift(window)) / df['close'].shift(window)) * 100


def calculate_williams_r(df, window=14):
    high_14 = df['high'].rolling(window=window).max()
    low_14 = df['low'].rolling(window=window).min()
    return (high_14 - df['close']) / (high_14 - low_14) * -100


def calculate_adx(df, window=14):
    df['+dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                         np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['-dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                         np.maximum(df['low'].shift(1) - df['low'], 0), 0)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    df['tr'] = np.maximum.reduce([tr1, tr2, tr3])
    df['+di'] = 100 * (df['+dm'].rolling(window=14).sum() / df['tr'].rolling(window=14).sum())
    df['-di'] = 100 * (df['-dm'].rolling(window=14).sum() / df['tr'].rolling(window=14).sum())
    df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
    return df['dx'].rolling(window=14).mean()


def create_lag_features(df, indicators=None, lags=5):
    """
    动态生成滞后特征。
    """
    indicators = indicators or ['close', 'rsi', 'macd_histogram']
    lag_features = []

    print("生成滞后特征...")
    for indicator in indicators:
        if indicator in df.columns:
            for lag in range(1, lags + 1):
                lag_features.append(df[indicator].shift(lag).rename(f'lag_{indicator}_{lag}'))
        else:
            print(f"警告：缺少 {indicator} 指标，跳过滞后特征生成。")

    # 合并滞后特征
    if lag_features:
        lag_df = pd.concat(lag_features, axis=1)
        df = pd.concat([df, lag_df], axis=1)

    # 删除含有 NaN 的行
    df.dropna(inplace=True)
    print("滞后特征生成完成。")
    return df
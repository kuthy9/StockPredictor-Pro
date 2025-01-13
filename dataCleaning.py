import pandas as pd
import os
import numpy as np

REQUIRED_COLUMNS = ["ts_code", "timestamp", "open", "high", "low", "close", "vol", "amount"]

def clean_data(input_file=None, ticker=None, config=None, debug=False):
    """清洗原始数据并保存清洗后的数据"""

    if ticker and config:
        # 通过ticker和config构造文件路径
        raw_data_dir = config["data_fetch"]["data_path"]["raw_data_path"]
        cleaned_data_dir = config['data_clean']['data_path']['cleaned_data_path']

        # 确保清洗后的数据路径存在
        os.makedirs(cleaned_data_dir, exist_ok=True)

        input_file = os.path.join(raw_data_dir, f"{ticker}_5y.parquet")


    if not os.path.exists(input_file):
        raise FileNotFoundError(f"未找到输入文件：{input_file}")

    df = pd.read_parquet(input_file)

    # 检查时间戳是否在索引中
    if isinstance(df.index, pd.DatetimeIndex):
        print("[DEBUG] 时间戳在索引中，将其重置为列")
        df.reset_index(inplace=True)

        # 显式重命名时间戳列
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'timestamp'}, inplace=True)
            print("[DEBUG] 索引列 'Date' 已重命名为 'timestamp'")

    # 检查是否存在 'timestamp' 列
    if 'timestamp' not in df.columns:
        raise KeyError("[错误] 数据中仍然没有 'timestamp' 列，请检查输入数据！")

    # 转换列名为小写
    original_columns = list(df.columns)
    df.columns = df.columns.str.lower()
    if debug:
        print(f"[调试] 原始列名: {original_columns}, 转换后列名: {list(df.columns)}")

    # 转换时间戳列
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if debug:
        print(f"[调试] 时间戳列转换完成: {df['timestamp'].head()}")

    # 删除重复时间戳
    df.sort_values(by='timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', inplace=True)

    # 填充缺失值
    if 'volume' not in df.columns and 'vol' in df.columns:
        df['volume'] = df['vol']
    if df.isnull().any().any():
        df.interpolate(method='linear', inplace=True)

    # 异常值处理
    for col in ['open', 'close', 'high', 'low', 'volume']:
        if col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # 验证列完整性
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        print(f"[警告] 缺少以下列: {missing_columns}")

    # 默认输出路径
    output_file = os.path.join(config['data_clean']['data_path']
                               ['cleaned_data_path'], f"cleaned_{ticker}_5y.parquet")

    # 保存清洗后的数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, index=False)
    if debug:
        print(f"[调试] 清洗后数据已保存至: {output_file}")

    return df  # 返回清洗后的数据

import os
import importlib.util

# 全局缓存变量
CONFIG_CACHE = None

# 配置内容
config = {

    # 目标股股票
    "ticker": "AAPL",


    # 全局参数
    "global": {
        "initial_capital": 1000000,
        "max_risk_percentage": 0.05,
        "slippage_rate": 0.001,  # 滑点比例
        "fee_rate": 0.001,
        "datetime_format": "%Y-%m-%d %H:%M:%S",
        "simulate_execution": True
    },


    # 数据抓取模块参数
    "data_fetch": {
        "start_date": "2018-01-01",
        "end_date": "2024-11-24",
        "source": "google_finance",
        "period": "5 Y",
        "interval": "1 D",
        "data_path": {
            "raw_data_path": "./data/raw/"
        },
        "retry_count":1
    },


    # 数据清洗模块参数
    "data_clean":{
        "data_path":{
            "cleaned_data_path": "./data/cleaned/",    
        }
    },


    # 网格参数
    "grid": {
        "grid_size": 5,          # 网格间距
        "min_capital_threshold": 100,
        "max_position_ratio": 0.8,
        "volatility_sensitivity": 0.5,
        "trend_threshold": 0.03,
        "rsi_overbought": 70,
        "rsi_oversold": 30
    },


    # 马丁格尔参数
    "martingale": {
        "initial_multiplier": 1.2,     # 马丁格尔加仓倍数
        "multiplier_step": 1.5, 
        "max_multiplier": 8, 
        "min_capital_threshold": 10000, 
        "max_position_ratio": 0.5, 
        "max_loss_count": 3, 
        "stop_loss_percent": 0.15, 
        "partial_profit_percent": 0.10, 
        "crash_threshold": 0.10
    },


    # 止损参数
    "stop_loss": {
        "stop_loss_factor": 0.9, # 止损系数
        "min_capital_threshold": 100,
        "max_position_ratio": 0.8
    },


    # 趋势跟踪参数
    "trend_following": {
        "short_window": 20, 
        "long_window": 50, 
        "overbought_threshold": 70,  # 超买阈值
        "oversold_threshold": 30,    # 超卖阈值
        "min_capital_threshold": 10000, 
        "max_position_ratio": 0.5, 
        "trend_strength_threshold": 0.02, 
        "position_adjustment_rate": 0.1, 
        "macd_fast": 12, 
        "macd_slow": 26, 
        "macd_signal": 9, 
        "volatility_threshold": 0.02
    },


    # 市场情绪模块参数
    "market_sentiment": {
        "api_keys": {
            "newsapi": {"enabled": True, "key": "34f9d62251ee4a3783b258ac797e2d80"},
            "alpha_vantage": {"enabled": True, "key": "3M4WTV88QABL4RBA"},
            "yahoo_finance": {"enabled": True}
        },
        "cache_duration_hours": 0,
        "deduplication_threshold": 0.8,
        "sentiment_thresholds": {
            "positive": 0.05,
            "negative": -0.05
        },
        "default_sentiment": {
            "compound": 0.0,
            "pos": 0.0,
            "neu": 1.0,
            "neg": 0.0,
            "sentiment_label": "neutral"
        },
        "relevance_filter": True,
        "time_range_days": 180,
        "max_delay_days": "2d",  # 新闻与股票时间戳对齐的最大延迟时间
        "data_path":{
            "sentiment_analysis_path": "./data/sentiment_analysis/"
        }
    },

    # 文件存储路径（待定）
    "data_paths": {
        "account_status_file": "./data/account_status.json",
        "indicators_path": "./data/indicators/{ticker}_indicators.parquet",
        "log_file_path": "./logs/trading.log"
    },

    # 预测模型参数
    "model_training": {
        "lstm": {
            "epochs": 100,
            "batch_size": 32,
            "patience": 5,
            "dropout_rate": 0.2,
            "lstm_units": [128, 64, 32]
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1
        },
        "self_learning": {
            "iterations": 5,
            "error_threshold_percentile": 90,
            "sample_weight_factor": 1.5
        },
        "hybrid_model": {
            "percentage": 0.8,
            "alpha_min": 0.3,
            "alpha_max": 0.7,
            "future_prediction_days": 300,
            "simulations": 15,
            "expand_factor": 1.2,
            "lags": 20,
            "self_learning": True
        },
        "data_path":{
            "prediction_path": "./data/predictions/{ticker}_predictions.parquet"
        }
    },


    # 风险监测模块参数
    "risk_management": {
        "risk_tolerance": 0.05,            # 风险容忍度（如5%）。
        "max_trade_quantity": 1000,        # 单次交易的最大数量限制。
        "volatility_threshold": 0.02,      # 波动性阈值，用于市场波动性校验。
        "risk_adjustment_factor": 0.5,     # 风险调整因子（例如市场波动时减少的交易量比例）。
    },


    # 实时交易模块参数
    "real_time_trading": {
        "max_retry_attempts": 3,       # 最大重试次数
        "retry_interval": 5           # 重试间隔时间（秒）
    },


    # 回测模块参数
    "backtest": {
    "max_drawdown_limit": 0.2,         # 最大回撤限制
    "sharpe_ratio_target": 1.5,        # 目标夏普比率
    "risk_levels": [0.01, 0.02, 0.05], # 风险水平列表
    "default_strategy": "GridStrategy", # 默认策略
    "parameter_grid": {                # 策略参数网格
        "grid_size": [5, 10, 15],
        "multiplier_step": [1.5, 2.0, 2.5]
        }
    }
}


def load_config(file_path=None):
    """
    加载配置文件，支持直接加载 Python 配置。
    """
    global CONFIG_CACHE

    if CONFIG_CACHE is not None:
        return CONFIG_CACHE  # 如果已加载，直接返回缓存

    if file_path is None or file_path.endswith(".py"):
        # 直接使用当前文件中的 `config` 字典
        CONFIG_CACHE = config
    else:
        print(f"[警告] 不支持的文件格式：{file_path}，请使用 Python 文件。")
        CONFIG_CACHE = {}

    preprocess_paths(CONFIG_CACHE)
    return CONFIG_CACHE


def preprocess_paths(config):
    """
    确保配置中的路径存在
    """
    paths = config.get("data_paths", {})
    for key, path in paths.items():
        if isinstance(path, str) and "{" not in path:  # 排除模板路径
            os.makedirs(os.path.dirname(path), exist_ok=True)


# 测试加载配置
if __name__ == "__main__":
    config_data = load_config()
    print("[配置加载成功] 以下为完整配置：")
    for key, value in config_data.items():
        print(f"{key}: {value}")
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from config import load_config
from dataFetch import fetch_historical_data, fetch_realtime_data
from dataCleaning import clean_data
from model import hybrid_model
from strategy import TransactionModule, StrategyModule, GridStrategy, MartingaleStrategy, StopLossStrategy
from riskMonitor import RiskManagementModule
from trade import RealTimeTradingModule
import pytz
from ib_insync import IB, Stock, util


def setup_logging():
    logging.basicConfig(filename='trading_log.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    setup_logging()
    # 加载配置文件
    config = load_config()
    ticker = config["ticker"]
    global_params = config["global"]

    # 初始化IB连接
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1, timeout=10)
    print("Connected:", ib.isConnected())

    accounts = ib.managedAccounts()
    print("Mode Account:", accounts)  
    if accounts:
        account_id = "DUA559603"  
        accountSummary = ib.accountSummary(account_id)
        for summary in accountSummary:
            print(f"{summary.tag}: {summary.value}")
    else:
        print("没有找到管理的账户。")


    transaction_module = TransactionModule(config)

    # 询问用户是否需要重置账户状态
    user_input = input("是否需要重置账户状态以开始新测试？(yes/no): ").strip().lower()
    if user_input == "yes":
        transaction_module.reset_account_status()
        # logging.info("账户状态已重置。")
        print("账户状态已重置。")
    elif user_input == "no":
        # logging.info("保留当前账户状态。")
        print("保留当前账户状态。")
    else:
        # logging.warning("输入无效，默认保留当前账户状态。")
        print ("输入无效，默认保留当前账户状态。")

    account_status = transaction_module.get_account_status()
    # logging.info(f"初始账户状态: {account_status}")
    print (f"初始账户状态: {account_status}")

    print(f"[主程序] 初始账户状态: {account_status}")

    # 初始化风险管理模块
    risk_management = RiskManagementModule(
        risk_config=config["risk_management"],
        global_params=config["global"],
        account_status=account_status,
        config=config,
        ib=ib
    )

    # 初始化实时交易模块
    trading_module = RealTimeTradingModule(
        trading_config=config["real_time_trading"],
        global_params=global_params,
        account_status=account_status,
        risk_management=risk_management,
        ib=ib
    )

    # 初始化策略模块
    strategies = [
        GridStrategy(**config["grid"], global_params=global_params),
        MartingaleStrategy(**config["martingale"], global_params=global_params),
        StopLossStrategy(**config["stop_loss"], global_params=global_params),
        # TrendFollowingStrategy(**config["trend_following"], global_params=global_params)
    ]
    strategy_module = StrategyModule(
        strategies=strategies,
        transaction_module=trading_module,
        risk_management=risk_management,
        global_config=global_params
    )

    # 数据处理和模型初始化
    try:
        historical_data = fetch_historical_data(ticker, ib, config)
        cleaned_data = clean_data(historical_data, ticker, config, debug=True)
        cleaned_data_path = f"./data/cleaned/cleaned_{ticker}_5y.parquet"
        cleaned_data.to_parquet(cleaned_data_path)
        hybrid_model(
            config=config,
            data_path=cleaned_data_path,
            lags=config["model_training"]["hybrid_model"].get("lags", 14),
            days_to_predict=config["model_training"]["hybrid_model"].get("future_prediction_days", 7),
            ticker=ticker,
            enable_self_learning=config["model_training"]["hybrid_model"].get("self_learning", True)
        )
    except Exception as e:
        # logging.error(f"数据处理或模型训练失败: {e}")
        print (f"数据处理或模型训练失败: {e}")
        return

    # 加载预测数据
    predictions = pd.read_parquet(config['model_training']['data_path']['prediction_path'].format(ticker=ticker))
    if not isinstance(predictions.index, pd.DatetimeIndex):
        predictions = predictions.set_index('timestamp')


    # 打印所有日期以检查
    print(predictions.index)  # 这应该显示为DatetimeIndex
    print(predictions.index.tz)  # 这应该显示时区信息


    # 设置中频交易间隔
    trading_interval = timedelta(minutes=2)  # 每2分钟交易一次
    last_trade_time = datetime.now()

    try:
        while True:
            current_time = datetime.now()
            if current_time - last_trade_time >= trading_interval:
                last_trade_time = current_time

                # logging.info(f"当前时间: {current_time}")
                print (f"当前时间: {current_time}")

                # 获取实时数据
                realtime_data = fetch_realtime_data(ticker, ib, config)
                if not realtime_data:
                    # logging.warning("无法获取实时数据，跳过此周期。")
                    print ("无法获取实时数据，跳过此周期。")
                    continue

                current_price = realtime_data["close"]
                # logging.info(f"当前价格: {current_price}")
                print (f"当前价格: {current_price}")


                # 转换到本地时区
                current_date = pd.Timestamp.now(tz='America/Toronto').date()
                # logging.info(f"当前日期: {current_date}")
                print (f"当前日期: {current_date}")
                current_date_ts = pd.Timestamp(current_date, tz='America/Toronto')


                # 检查是否有当日预测数据，如果没有则使用最近的预测
                if current_date_ts in predictions.index:
                    pred_data = predictions.loc[current_date_ts]
                    print(f"成功获取预测数据: {pred_data}")
                    predicted_price, lower_ci, upper_ci = pred_data["predicted_price"], pred_data["lower_ci"], pred_data["upper_ci"]
                    # logging.info(f"预测数据: {pred_data}")
                    print(f"预测数据: {pred_data}")

                    # 调用策略模块获取交易指令
                    action = strategy_module.execute(current_price, predicted_price, lower_ci, upper_ci, cleaned_data)
                    # logging.info(f"策略执行结果: {action}")
                    print (f"策略执行结果: {action}")

                    if action:
                        risk_management.adjust_risk_tolerance()
                        action_data = risk_management.check_risk(action, ticker)
                        # logging.info(f"风险检查结果: {action_data}")
                        print (f"风险检查结果: {action_data}")
                        
                        if action_data["status"] == 'approved':
                            trading_module.execute_trade(action_data, ticker)
                            # logging.info(f"交易执行成功: {action_data}")
                            print (f"交易执行成功: {action_data}")
                        else:
                            # logging.info(f"交易被阻止: {action_data['reason']}")
                            print (f"交易被阻止: {action_data['reason']}")
                    else:
                        # logging.info("没有生成有效的交易指令。")
                        print ("没有生成有效的交易指令。")

                else:
                    # 如果没有当日预测数据，使用最近的预测
                    print("日期匹配失败，尝试使用最近的预测数据")
                    nearest_date = predictions.index.get_loc(current_date, method='nearest')
                    pred_data = predictions.iloc[nearest_date]
                    print(f"无当日预测数据，使用最近的预测数据: {pred_data}")
                    
                    # 使用最近的预测数据进行交易策略的计算
                    action = strategy_module.execute(current_price, pred_data["predicted_price"], 
                                                    pred_data["lower_ci"], pred_data["upper_ci"], 
                                                    cleaned_data)
                    print(f"策略执行结果（使用最近预测）: {action}")

                    if action:
                        risk_management.adjust_risk_tolerance()
                        action_data = risk_management.check_risk(action, ticker)
                        print(f"风险检查结果（使用最近预测）: {action_data}")
                        
                        if action_data["status"] == 'approved':
                            trading_module.execute_trade(action_data, ticker)
                            print(f"交易执行成功（使用最近预测）: {action_data}")
                        else:
                            print(f"交易被阻止（使用最近预测）: {action_data['reason']}")
                    else:
                        print("没有生成有效的交易指令（使用最近预测）。")

                # 交易总结
                summary = trading_module.trade_summary()
                # logging.info(f"交易总结: {summary}")
                print (f"交易总结: {summary}")

                print(f"[主程序] 交易总结: {summary}")

            time.sleep(10)  # 防止CPU过度使用

    except KeyboardInterrupt:
        # logging.info("用户终止了交易程序。")
        print ("用户终止了交易程序。")
    except Exception as e:
        # logging.error(f"程序运行过程中发生错误: {e}")
        print (f"程序运行过程中发生错误: {e}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from dataFetch import fetch_historical_data
from ib_insync import IB, Stock, util


class RiskManagementModule:
    def __init__(self, risk_config, global_params, account_status, config, ib=None):
        """
        初始化风险监测模块
        :param risk_config: 风险管理相关配置
        :param global_params: 全局参数
        :param account_status: 当前账户状态
        """
        if ib is None:
            self.ib = IB()
            self.ib.connect('127.0.0.1', 7497, clientId=1)  
        else:
            self.ib = ib

        self.risk_config = risk_config
        self.global_params = global_params
        self.account_status = account_status
        self.config = config  
        self.max_risk_percentage = risk_config['risk_tolerance']
        self.max_trade_quantity = risk_config['max_trade_quantity']
        self.fee_rate = global_params['fee_rate']
        self.initial_capital = global_params['initial_capital']
        self.volatility_threshold = risk_config.get('volatility_threshold', 0.02)  # 默认值 2%
        self.risk_adjustment_factor = risk_config.get('risk_adjustment_factor', 0.5)  # 默认调整因子 0.5

    def calculate_risk(self, action_data):
        action = action_data['action']
        quantity = action_data['quantity']
        price = action_data['price']
        return quantity * price if action == 'buy' else -quantity * price

    def assess_market_volatility(self, ticker):
        """
        评估市场波动性，基于ATR或标准差。
        :param historical_data: 包含历史价格数据的 DataFrame
        :return: 波动性评分（高/低）
        """
        # 获取历史数据
        historical_data = fetch_historical_data(ticker, self.ib, config=self.config, period="14 D", interval="1 D")

        # print("[调试] historical_data 信息:")
        # print(f"Columns: {historical_data.columns}")
        # print(f"Shape: {historical_data.shape}")
        # print(f"Data preview:\n{historical_data.head()}")
        # print(f"NaN 数量:\n{historical_data.isna().sum()}")

        if historical_data.empty or not {'High', 'Low', 'Close'}.issubset(
            historical_data.columns) or len(historical_data) < 14:
            print(f"[风险模块] 无法获取有效的 {ticker} 历史数据进行波动性评估")
            return "Low"

        try:
            # ATR计算
            high_low = historical_data['High'] - historical_data['Low']
            high_close = abs(historical_data['High'] - historical_data['Close'].shift(1).fillna(0))  # 使用fillna处理第一行的NaN
            low_close = abs(historical_data['Low'] - historical_data['Close'].shift(1).fillna(0))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # 检查TR是否足够长
            if len(tr) < 14:
                print(f"[风险模块] 数据不足以计算ATR，返回低波动性")
                return "Low"

            atr = tr.rolling(window=14, min_periods=14).mean()  # 使用min_periods确保只有5个数据点时才开始计算

            # 波动率阈值判断
            if atr.iloc[-1] > self.volatility_threshold * historical_data['Close'].iloc[-1]:
                return "High"
            return "Low"
        except Exception as e:
            print(f"[风险模块] 波动性评估过程中发生错误: {e}")
            return "Low"


    def check_risk(self, action_data, ticker):
        """
        核心风险校验逻辑。
        :param action_data: 初步交易指令
        :return: 通过校验的指令或阻止交易的理由
        """
        try:
            # 如果没有有效的 action_data，直接返回被阻止的状态
            if not action_data:
                return {"status": "blocked", "reason": "没有交易指令"}

            # 核心字段校验
            required_keys = {"action", "quantity", "price", "strategy"}
            missing_keys = required_keys - action_data.keys()
            if missing_keys:
                return {"status": "blocked", "reason": f"交易指令缺少字段: {', '.join(missing_keys)}"}

            action_data['ticker'] = ticker

            # 更新账户状态
            self.update_account_status()

            capital = self.account_status['capital']
            position_value = self.account_status['position'] * action_data['price']
            trade_risk = self.calculate_risk(action_data)
            total_assets = capital + position_value

            # 检查风险敞口
            new_risk = (position_value + trade_risk) / total_assets if total_assets > 0 else 0
            if new_risk > self.max_risk_percentage:
                max_trade_value = self.max_risk_percentage * total_assets - position_value
                max_quantity = max_trade_value // action_data['price']
                if max_quantity > 0:
                    action_data['quantity'] = min(action_data['quantity'], max_quantity)
                    print(f"[调试信息] 风险监测：调整交易量为 {action_data['quantity']} 以符合风险容忍度")
                else:
                    min_trade_quantity = 10  # 假设最小交易量为10股
                    if action_data['quantity'] > min_trade_quantity:
                        action_data['quantity'] = min_trade_quantity
                        print(f"[调试信息] 风险监测：交易量调整为最小交易量 {action_data['quantity']} 以符合风险容忍度")
                    else:
                        print("[调试信息] 风险监测：交易超出风险敞口范围，已被阻止")
                        return {'status': 'blocked', 'reason': '交易超出风险敞口范围'}
                    
            # 实时市场数据更新
            market_data = self.ib.reqMktData(Stock(ticker, 'SMART', 'USD'))
            self.ib.sleep(1)  # 等待数据更新
            current_price = market_data.marketPrice() or action_data['price']  # 使用实时价格或指令中的价格

            # 更新交易指令的价格
            action_data['price'] = current_price

            # 检查市场波动性
            volatility = self.assess_market_volatility(action_data['ticker'])
            if volatility == "high":
                if action_data['quantity'] > 1:  # 确保调整后的数量合理
                    action_data['quantity'] //= 2
                    print(f"[调试信息] 风险监测：市场波动性较高，降低交易量为 {action_data['quantity']}")
                else:
                    print("[调试信息] 风险监测：市场波动性过高，交易被取消")
                    return {'status': 'blocked', 'reason': '市场波动性过高，取消交易'}
                
            # 处理异常交易，例如连续亏损或超出风险范围。
            if self.account_status['capital'] < 0:
                print("[警告] 账户资金为负，暂停交易！")
                return {'status': 'blocked', 'reason': '账户资金不足'}

            # 如果交易通过所有风险校验
            print("[调试信息] 风险监测：交易指令通过风险校验")
            # 返回调整后的交易指令
            return {"status": "approved", "reason": "交易通过风险监测", **action_data}
        
        except Exception as e:
            print(f"[风险模块] 检查过程中出现异常：{e}")
            return {"status": "blocked", "reason": f"内部错误: {e}"}


    def adjust_risk_tolerance(self):
        # 保持不变，但可以考虑根据市场实时数据调整
        base_tolerance = self.initial_capital * 0.01
        capital_ratio = self.account_status['capital'] / self.initial_capital
        self.max_risk_percentage = max(min(base_tolerance * capital_ratio, 0.10), 0.01)

        # 根据市场波动性调整
        volatility = self.assess_market_volatility(self.config["ticker"])  # 使用默认ticker或从配置中获取
        if volatility == "High":
            self.max_risk_percentage *= (1 - self.risk_adjustment_factor)
        else:
            self.max_risk_percentage *= (1 + self.risk_adjustment_factor)
        
        print(f"[调试信息] 风险监测：动态调整风险容忍度为 {self.max_risk_percentage:.2%}")


    def update_account_status(self):
        """
        从IBKR获取并更新账户状态
        """
        account_values = self.ib.accountValues()
        for value in account_values:
            if value.tag == 'NetLiquidation':
                self.account_status['capital'] = float(value.value)
            elif value.tag == 'GrossPositionValue':
                self.account_status['position'] = float(value.value)  # 注意：这可能需要更复杂的逻辑来处理多个持仓
        print(f"[调试信息] 风险监测：账户状态已更新，资本：{self.account_status['capital']:.2f}, 持仓价值：{self.account_status['position']:.2f}")

import random
from techlag_index import calculate_technical_indicators
import os
import pandas as pd
from datetime import datetime, timedelta


# 考虑手续费和滑点问题
class TransactionModule:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config["global"]["initial_capital"]
        self.account_status = {'capital': self.initial_capital, 'position': 0}
        self.slippage_rate = config['global']['slippage_rate']
        self.fee_rate = config['global']['fee_rate']
        self.trade_log = []  # 交易日志

    def get_account_status(self):
        return self.account_status
    
    def reset_account_status(self):
        self.account_status = {'capital': self.initial_capital, 'position': 0}
        print("[信息] 账户状态已成功重置。")

    def apply_slippage_and_fees(self, action):
        price = action['price']
        quantity = action['quantity']
        action_type = action['action']

        if action_type == 'buy':
            adjusted_price = price * (1 + self.slippage_rate)
        elif action_type == 'sell':
            adjusted_price = price * (1 - self.slippage_rate)
        else:
            raise ValueError(f"未知的交易类型: {action_type}")

        transaction_cost = adjusted_price * quantity * self.fee_rate
        return {'action': action_type, 'quantity': quantity, 'price': adjusted_price, 'cost': transaction_cost}

    def execute_trade(self, action, ticker):
        price = action['price']
        quantity = action['quantity']
        action_type = action['action']

        adjusted_action = self.apply_slippage_and_fees(action)
        total_cost = adjusted_action['price'] * quantity + adjusted_action['cost']

        if action_type == 'buy':
            if total_cost > self.account_status['capital']:
                print(f"[错误] 可用资金不足，无法完成交易: {total_cost:.2f} > {self.account_status['capital']:.2f}")
                return
            self.account_status['capital'] -= total_cost
            self.account_status['position'] += quantity
        elif action_type == 'sell':
            if quantity > self.account_status['position']:
                quantity = self.account_status['position']  # 避免卖出超过持仓数量
            total_revenue = adjusted_action['price'] * quantity - adjusted_action['cost']
            self.account_status['capital'] += total_revenue
            self.account_status['position'] -= quantity

        # 记录交易日志
        self.trade_log.append({
            'ticker': ticker,
            'action': action_type,
            'quantity': quantity,
            'price': adjusted_action['price'],
            'timestamp': datetime.now(),
            'cost': adjusted_action['cost'],
            'capital_after': self.account_status['capital'],
            'position_after': self.account_status['position']
        })

        print(f"[交易完成] {action_type.upper()} -> 数量: {quantity}, 价格: {adjusted_action['price']:.2f}")
        print(f"[账户状态] -> 资金: {self.account_status['capital']:.2f}, 持仓: {self.account_status['position']}")

    def get_trade_log(self):
        return self.trade_log


# 通用资金和持仓检查
class BaseStrategy:
    def __init__(self, global_params):
        self.slippage_rate = global_params['slippage_rate']
        self.fee_rate = global_params['fee_rate']

    def check_funds_and_position(self, current_price, account_status, min_capital_threshold, max_position_ratio):
        total_cost = current_price * (1 + self.slippage_rate) * (1 + self.fee_rate)
        if account_status['capital'] - total_cost < min_capital_threshold:
            print(f"[{self.__class__.__name__}] 资金不足，无法完成交易")
            return False
        max_position = int(account_status['capital'] * max_position_ratio / current_price)
        if abs(account_status['position']) > max_position:  # 使用abs处理卖空位置
            print(f"[{self.__class__.__name__}] 持仓已达到限制，无法买入或卖出更多股票")
            return False
        return True


class GridStrategy:
    def __init__(self, grid_size, global_params, min_capital_threshold, max_position_ratio, 
                 volatility_sensitivity=0.5, trend_threshold=0.03, rsi_overbought=70, rsi_oversold=30,
                 time_based_stop_loss=1, profit_taking_percent=0.05, crash_threshold=0.10):
        self.grid_size = grid_size
        self.global_params = global_params
        self.min_capital_threshold = min_capital_threshold
        self.max_position_ratio = max_position_ratio
        self.volatility_sensitivity = volatility_sensitivity
        self.trend_threshold = trend_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.slippage_rate = global_params['slippage_rate']
        self.fee_rate = global_params['fee_rate']
        self.time_based_stop_loss = timedelta(days=time_based_stop_loss)  # 持仓时间止损
        self.profit_taking_percent = profit_taking_percent  # 部分止盈比例
        self.crash_threshold = crash_threshold  # 崩盘阈值，设为10%的跌幅

        # 存储交易信息以用于后续的风险控制
        self.trades = {}  # 字典存储交易信息，key为交易id，value为交易详情

    def adjust_grid_size(self, indicators):
        atr = indicators.get('atr', 0).iloc[-1]
        adjusted_grid_size = self.grid_size * (1 + self.volatility_sensitivity * atr / self.grid_size)
        return adjusted_grid_size

    def is_in_trend(self, indicators):
        sma_5 = indicators.get('sma_5', indicators.get('close', 0)).iloc[-1]
        sma_50 = indicators.get('sma_50', indicators.get('close', 0)).iloc[-1]

        # 如果sma_5和sma_50是Series，我们需要比较它们的当前值或平均值
        if isinstance(sma_5, pd.Series) and isinstance(sma_50, pd.Series):
            # 仅考虑最后一个值（即最新的数据点）
            trend_diff = abs(sma_5.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        else:
            # 如果不是Series，则直接比较
            trend_diff = abs(sma_5 - sma_50) / sma_50

        return trend_diff > self.trend_threshold

    def check_funds_and_position(self, current_price, account_status):
        total_cost = current_price * (1 + self.slippage_rate) * (1 + self.fee_rate)
        if account_status['capital'] - total_cost < self.min_capital_threshold:
            print("[GridStrategy] 资金不足，无法完成交易")
            return False
        max_position = int(account_status['capital'] * self.max_position_ratio / current_price)
        if account_status['position'] > max_position:
            print("[GridStrategy] 持仓已达到限制，无法买入更多股票")
            return False
        return True

    def check_for_time_based_stop_loss(self, current_time, trade_id, current_price):
        if trade_id in self.trades:
            trade = self.trades[trade_id]
            # 确保current_time和trade['timestamp']都是datetime对象
            if not isinstance(current_time, datetime):
                current_time = datetime.fromtimestamp(current_time)
            if not isinstance(trade['timestamp'], datetime):
                trade['timestamp'] = datetime.fromtimestamp(trade['timestamp'])
            if current_time - trade['timestamp'] > self.time_based_stop_loss:
                if trade['action'] == 'buy' and current_price < trade['price']:
                    return True, "时间止损"
        return False, None

    def check_for_profit_taking(self, trade_id, current_price):
        if trade_id in self.trades:
            trade = self.trades[trade_id]
            if trade['action'] == 'buy':
                profit_threshold = trade['price'] * (1 + self.profit_taking_percent)
                if current_price >= profit_threshold:
                    return True, int(trade['quantity'] * self.profit_taking_percent)  # 卖出部分头寸
        return False, None

    def check_for_market_crash(self, historical_data):
        # 这里我们假设最近的价格数据在historical_data的最后
        if len(historical_data) > 1:
            recent_price = historical_data.iloc[-1]['close']
            previous_price = historical_data.iloc[-2]['close']
            if (previous_price - recent_price) / previous_price > self.crash_threshold:
                return True
        return False

    def evaluate(self, current_price, predicted_price, lower_ci, upper_ci, account_status, historical_data):
        current_time = datetime.now()
        indicators = calculate_technical_indicators(historical_data, indicators=['rsi', 'sma_5', 'sma_50', 'atr', 'bollinger'])
        
        if not self.check_funds_and_position(current_price, account_status):
            return None

        adjusted_grid_size = self.adjust_grid_size(indicators)
        quantity = int(account_status['capital'] / (current_price * (1 + self.slippage_rate) * (1 + self.fee_rate)))

        # 确保数量至少为1
        quantity = max(1, quantity)  # 如果计算出的数量小于1，则设为1
        
        rsi = indicators.get('rsi', 50)
        bollinger_lower = indicators.get('lower_band', current_price)
        bollinger_upper = indicators.get('upper_band', current_price)

        # 检查市场崩盘
        if self.check_for_market_crash(historical_data):
            return {'action': 'sell_all', 'reason': '市场崩盘'}

        # 检查是否在趋势中
        if self.is_in_trend(indicators):
            return None
        
        # 检查持仓的止损和部分止盈情况
        for trade_id in list(self.trades.keys()):  # 我们使用 list() 来避免字典改变大小的问题
            time_stop_loss, reason = self.check_for_time_based_stop_loss(current_time, trade_id, current_price)
            if time_stop_loss:
                return {'action': 'sell', 'quantity': self.trades[trade_id]['quantity'], 'price': current_price, 'reason': reason}

            profit_take, quantity_to_sell = self.check_for_profit_taking(trade_id, current_price)
            if profit_take:
                self.trades[trade_id]['quantity'] -= quantity_to_sell  # 更新剩余持仓
                return {'action': 'sell', 'quantity': quantity_to_sell, 'price': current_price, 'reason': '部分止盈'}

        if current_price % adjusted_grid_size < adjusted_grid_size / 2:
            # 检查RSI是否处于超卖状态，同时价格低于下布林带
            if rsi.iloc[-1] <= self.rsi_overbought and current_price < bollinger_lower.iloc[-1]:
                trade_id = len(self.trades) + 1
                self.trades[trade_id] = {"action": "buy", "quantity": quantity, "price": current_price, "timestamp": current_time}
                return {'action': 'buy', 'quantity': quantity, 'price': current_price}
        elif current_price % adjusted_grid_size > adjusted_grid_size / 2 and account_status['position'] > 0:
            # 检查RSI是否处于超买状态，同时价格高于上布林带
            if rsi.iloc[-1] >= self.rsi_oversold and current_price > bollinger_upper.iloc[-1]:
                # 这里可以根据需要卖出全部或部分持仓
                return {'action': 'sell', 'quantity': account_status['position'], 'price': current_price}
        
        return None



class MartingaleStrategy:
    def __init__(self, initial_multiplier, multiplier_step, max_multiplier, global_params, 
                 min_capital_threshold, max_position_ratio, max_loss_count, 
                 stop_loss_percent, partial_profit_percent, crash_threshold):
        """
        :param initial_multiplier: 初始交易量的倍数
        :param multiplier_step: 每次损失后的倍增因子
        :param max_multiplier: 最大允许的倍增因子
        :param global_params: 全局参数，包含滑点和费率等
        :param min_capital_threshold: 最小资金阈值，低于这个值时停止交易
        :param max_position_ratio: 最大持仓比例
        :param max_loss_count: 最大连续损失次数
        :param stop_loss_percent: 止损百分比
        :param partial_profit_percent: 部分盈利锁定的百分比
        :param crash_threshold: 市场崩盘的阈值，设为10%的跌幅
        """
        self.initial_multiplier = initial_multiplier
        self.multiplier = initial_multiplier
        self.multiplier_step = multiplier_step
        self.max_multiplier = max_multiplier
        self.global_params = global_params
        self.slippage_rate = global_params['slippage_rate']
        self.fee_rate = global_params['fee_rate']
        self.min_capital_threshold = min_capital_threshold
        self.max_position_ratio = max_position_ratio
        self.max_loss_count = max_loss_count
        self.stop_loss_percent = stop_loss_percent
        self.partial_profit_percent = partial_profit_percent
        self.crash_threshold = crash_threshold
        self.loss_count = 0
        self.last_trade_price = None
        self.last_trade_action = None

    def check_funds_and_position(self, current_price, account_status, multiplier):
        total_cost = current_price * (1 + self.slippage_rate) * (1 + self.fee_rate) * multiplier
        if account_status['capital'] - total_cost < self.min_capital_threshold:
            print(f"[MartingaleStrategy] 资金不足，无法完成交易，当前资金：{account_status['capital']}")
            return False
        max_position = int(account_status['capital'] * self.max_position_ratio / current_price)
        if account_status['position'] > max_position:
            print(f"[MartingaleStrategy] 持仓已达到限制，无法买入更多股票")
            return False
        return True

    def check_for_market_crash(self, historical_data):
        if len(historical_data) > 1:
            recent_price = historical_data.iloc[-1]['close']
            previous_price = historical_data.iloc[-2]['close']
            if (previous_price - recent_price) / previous_price > self.crash_threshold:
                return True
        return False

    def evaluate(self, current_price, predicted_price, lower_ci, upper_ci, account_status, historical_data):
        indicators = calculate_technical_indicators(historical_data, indicators=['rsi', 'atr'])
        
        # 确保atr是一个单一值
        atr = indicators.get('atr', 0).iloc[-1]

        # 检查市场崩盘情况
        if self.check_for_market_crash(historical_data):
            print("[MartingaleStrategy] 市场崩盘，触发紧急止损")
            return {'action': 'sell_all', 'reason': '市场崩盘'}

        if not self.check_funds_and_position(current_price, account_status, self.multiplier):
            return None

        dynamic_multiplier = self.multiplier * (1 + atr / 100)  # 动态调整倍增因子

        # 买入逻辑
        if predicted_price < current_price and account_status['capital'] > current_price:
            if self.last_trade_action == 'buy' and current_price < self.last_trade_price:
                self.loss_count += 1
            else:
                self.loss_count = 0

            if self.loss_count < self.max_loss_count:
                quantity = int((account_status['capital'] / (current_price * (1 + self.slippage_rate) * (1 + self.fee_rate))) * dynamic_multiplier)
                self.multiplier = min(self.multiplier * self.multiplier_step, self.max_multiplier)
                self.last_trade_price = current_price
                self.last_trade_action = 'buy'
                return {'action': 'buy', 'quantity': quantity, 'price': current_price}
            else:
                print("[MartingaleStrategy] 达到最大连续损失次数，停止交易")
                return None

        # 卖出逻辑
        elif predicted_price > current_price and account_status['position'] > 0:
            if self.last_trade_action == 'sell' and current_price > self.last_trade_price:
                self.loss_count += 1
            else:
                self.loss_count = 0

            if self.loss_count < self.max_loss_count:
                # 检查是否达到止损点
                if self.last_trade_price and current_price <= self.last_trade_price * (1 - self.stop_loss_percent):
                    print("[MartingaleStrategy] 触发止损")
                    return {'action': 'sell', 'quantity': account_status['position'], 'price': current_price}

                # 部分盈利锁定
                if current_price > self.last_trade_price * (1 + self.partial_profit_percent):
                    partial_quantity = int(account_status['position'] * self.partial_profit_percent)
                    if partial_quantity > 0:
                        print("[MartingaleStrategy] 部分盈利锁定")
                        return {'action': 'sell', 'quantity': partial_quantity, 'price': current_price}

                self.last_trade_price = current_price
                self.last_trade_action = 'sell'
                return {'action': 'sell', 'quantity': account_status['position'], 'price': current_price}
            else:
                print("[MartingaleStrategy] 达到最大连续损失次数，停止交易")
                return None

        return None

class StopLossStrategy(BaseStrategy):
    def __init__(self, stop_loss_factor, global_params, min_capital_threshold, max_position_ratio):
        self.stop_loss_factor = stop_loss_factor
        self.slippage_rate = global_params['slippage_rate']
        self.fee_rate = global_params['fee_rate']
        self.min_capital_threshold = min_capital_threshold
        self.max_position_ratio = max_position_ratio

    def evaluate(self, current_price, predicted_price, lower_ci, upper_ci, account_status, indicators=None, historical_data=None):
        stop_loss_price = lower_ci * self.stop_loss_factor
        if not self.check_funds_and_position(current_price, account_status, self.min_capital_threshold, self.max_position_ratio):
            return None

        # 卖出逻辑：触发止损
        if current_price < stop_loss_price and account_status['position'] > 0:
            return {'action': 'sell', 'quantity': account_status['position'], 'price': current_price}

        return None
    

# class TrendFollowingStrategy:
#     def __init__(self, short_window, long_window, overbought_threshold, oversold_threshold, 
#                  global_params, min_capital_threshold, max_position_ratio, 
#                  trend_strength_threshold, position_adjustment_rate, 
#                  macd_fast, macd_slow, macd_signal, volatility_threshold):
#         """
#         :param short_window: 短期移动平均线的时间窗口
#         :param long_window: 长期移动平均线的时间窗口
#         :param overbought_threshold: RSI超买阈值
#         :param oversold_threshold: RSI超卖阈值
#         :param global_params: 全局参数，包含滑点和费率等
#         :param min_capital_threshold: 最小资金阈值，低于这个值时停止交易
#         :param max_position_ratio: 最大持仓比例
#         :param trend_strength_threshold: 趋势强度阈值，用于调整头寸
#         :param position_adjustment_rate: 调整头寸的速率
#         :param macd_fast: MACD快速EMA
#         :param macd_slow: MACD慢速EMA
#         :param macd_signal: MACD信号线EMA
#         :param volatility_threshold: 波动性阈值，用于判断是否保持跟随
#         """
#         self.short_window = short_window
#         self.long_window = long_window
#         self.overbought_threshold = overbought_threshold
#         self.oversold_threshold = oversold_threshold
#         self.global_params = global_params
#         self.slippage_rate = global_params['slippage_rate']
#         self.fee_rate = global_params['fee_rate']
#         self.min_capital_threshold = min_capital_threshold
#         self.max_position_ratio = max_position_ratio
#         self.trend_strength_threshold = trend_strength_threshold
#         self.position_adjustment_rate = position_adjustment_rate
#         self.macd_fast = macd_fast
#         self.macd_slow = macd_slow
#         self.macd_signal = macd_signal
#         self.volatility_threshold = volatility_threshold
#         self.last_position = 0

#     def check_funds_and_position(self, current_price, account_status):
#         total_cost = current_price * (1 + self.slippage_rate) * (1 + self.fee_rate)
#         if account_status['capital'] - total_cost < self.min_capital_threshold:
#             print(f"[TrendFollowingStrategy] 资金不足，无法完成交易，当前资金：{account_status['capital']}")
#             return False
#         max_position = int(account_status['capital'] * self.max_position_ratio / current_price)
#         if abs(account_status['position']) > max_position:
#             print(f"[TrendFollowingStrategy] 持仓已达到限制，无法买入或卖出更多股票")
#             return False
#         return True

#     def calculate_macd(self, close):
#         ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
#         ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
#         macd_line = ema_fast - ema_slow
#         signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
#         histogram = macd_line - signal_line
#         return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]  # 返回单一值

#     def evaluate(self, current_price, predicted_price, lower_ci, upper_ci, account_status, historical_data):
#         indicators = calculate_technical_indicators(historical_data, indicators=['sma', 'ema', 'rsi', 'atr', 'macd'])
        
#         if not self.check_funds_and_position(current_price, account_status):
#             return None

#         # 获取所需的指标值
#         sma_data = indicators.get('sma', indicators.get('close', 0))
#         sma_short = sma_data.iloc[-self.short_window:].mean()
#         sma_long = sma_data.iloc[-self.long_window:].mean()
        
#         rsi = indicators.get('rsi', 50).iloc[-1]  # 获取最新RSI值
#         atr = indicators.get('atr', 0).iloc[-1]  # 获取最新ATR值
        
#         # MACD计算
#         macd_line, signal_line, histogram = self.calculate_macd(historical_data['close'])

#         # 趋势确认
#         trend_strength = abs(sma_short - sma_long) / sma_long
#         trend_up = sma_short > sma_long
#         trend_down = sma_short < sma_long

#         # 动态止损计算
#         stop_loss = current_price - (atr * 2) if trend_up else current_price + (atr * 2)

#         # 波动性计算
#         recent_volatility = atr / historical_data['close'].iloc[-1]  # 假设用ATR来衡量波动性

#         # 头寸调整
#         position_change = int(account_status['capital'] * self.position_adjustment_rate / current_price)

#         # 交易逻辑
#         if trend_up:
#             if rsi < self.overbought_threshold and histogram[-1] > histogram[-2]:  # MACD直方图上升作为领先指标
#                 if account_status['position'] < self.last_position:  # 增加头寸
#                     return {'action': 'buy', 'quantity': position_change, 'price': current_price}
#                 elif account_status['position'] > 0 and current_price < stop_loss:  # 止损
#                     return {'action': 'sell', 'quantity': account_status['position'], 'price': current_price, 'reason': '动态止损'}
#             else:
#                 if recent_volatility < self.volatility_threshold:  # 如果波动性在一定范围内，保持持仓
#                     return None  # 维持现状
#         elif trend_down:
#             if rsi > self.oversold_threshold and histogram[-1] < histogram[-2]:  # MACD直方图下降作为领先指标
#                 if account_status['position'] > self.last_position:  # 增加头寸
#                     return {'action': 'sell', 'quantity': position_change, 'price': current_price}
#                 elif account_status['position'] < 0 and current_price > stop_loss:  # 止损
#                     return {'action': 'buy', 'quantity': -account_status['position'], 'price': current_price, 'reason': '动态止损'}
#             else:
#                 if recent_volatility < self.volatility_threshold:  # 如果波动性在一定范围内，保持持仓
#                     return None  # 维持现状
#         else:  # 无明显趋势，减少头寸
#             if account_status['position'] != 0 and trend_strength < self.trend_strength_threshold:
#                 quantity_to_adjust = int(abs(account_status['position']) * self.position_adjustment_rate)
#                 return {'action': 'sell' if account_status['position'] > 0 else 'buy', 
#                         'quantity': quantity_to_adjust, 
#                         'price': current_price, 
#                         'reason': '无明显趋势，减少头寸'}

#         # 更新上一次持仓
#         self.last_position = account_status['position']
        
#         return None



class StrategyModule:
    def __init__(self, strategies, transaction_module, risk_management, global_config):
        self.strategies = strategies
        self.transaction_module = transaction_module
        self.risk_management = risk_management
        self.global_config = global_config

    def execute(self, current_price, predicted_price, lower_ci, upper_ci, historical_data):
        # 获取账户状态
        account_status = self.transaction_module.get_account_status()
        
        # 计算所有可能需要的技术指标
        indicators = calculate_technical_indicators(historical_data, indicators=['rsi', 'sma', 'ema', 'atr', 'macd', 'bollinger'])
        
        valid_actions = []
        for strategy in self.strategies:
            try:
                action = strategy.evaluate(
                    current_price=current_price,
                    predicted_price=predicted_price,
                    lower_ci=lower_ci,
                    upper_ci=upper_ci,
                    account_status=account_status,
                    historical_data=historical_data  # 传递完整的历史数据而不是仅传递indicators
                )
                if action:
                    action["strategy"] = strategy.__class__.__name__  # 添加策略名称
                    valid_actions.append(action)
            except Exception as e:
                print(f"[{strategy.__class__.__name__}] 策略执行时发生错误: {str(e)}")

        # 如果有多个有效的交易信号，可能需要一个方法来选择或组合这些信号
        if valid_actions:
            # 使用最佳策略选择逻辑
            return self.select_best_action(valid_actions, current_price)
        
        # 如果所有策略都没有生成有效指令，则返回 None
        return None

    def select_best_action(self, actions, current_price):
        # 定义一个简单的评分系统，基于预期收益和风险
        def action_score(action, current_price):
            # 假设买入的潜在收益和风险评估
            if action['action'] == 'buy':
                expected_return = (action.get('predicted_price', current_price) - current_price) / current_price
                risk = 0.1  # 假设一个固定的风险值，可以根据策略更精确地计算
                return expected_return / risk  # 简单地用收益率除以风险作为评分

            # 假设卖出的潜在收益和风险评估
            elif action['action'] == 'sell':
                expected_return = (current_price - action.get('predicted_price', current_price)) / current_price
                risk = 0.1  # 同样假设一个固定的风险值
                return expected_return / risk

            # 如果动作既不是买入也不是卖出，则返回一个较低的分数
            return -1

        # 评估每个动作并选择分数最高的
        scored_actions = [(action, action_score(action, current_price)) for action in actions]
        best_action = max(scored_actions, key=lambda x: x[1], default=(None, -float('inf')))[0]

        # 如果没有有效的动作（虽然这应该不会发生，因为我们已经有valid_actions）
        if best_action is None:
            return None

        return best_action
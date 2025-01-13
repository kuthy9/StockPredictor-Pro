import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
import random
from config import load_config
from strategy import StrategyModule
from model import hybrid_model

class BacktestModule:
    def __init__(self, config, initial_capital):
        """
        初始化回测模块
        :param config: 配置文件
        :param initial_capital: 初始资金
        """
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.trade_log = []
        self.performance_metrics = []
        self.strategy_combinations = []
        self.risk_levels = config['backtest']['risk_levels']
        self.fee_rate = config['global']['fee_rate']
        self.slippage_rate = config['global']['slippage_rate']

    def reset_backtest(self):
        """
        重置回测状态以支持多次模拟。
        """
        self.current_capital = self.initial_capital
        self.position = 0
        self.trade_log = []

    def get_returns(self):
        """
        提取交易日志中的收益。
        :return: 收益列表。
        """
        return [trade.get('profit', 0) for trade in self.trade_log 
                if trade['action'] == 'sell']

    def execute_trade(self, trade):
        """
        模拟交易执行
        :param trade: 交易信号
        """
        action = trade['action']
        quantity = trade['quantity']
        price = trade['price']
        timestamp = trade.get('timestamp', datetime.now())
        fee_rate = self.config['global']['fee_rate']

        if action == 'buy':
            cost = quantity * price * (1 + fee_rate)
            if self.current_capital >= cost:
                self.current_capital -= cost
                self.position += quantity
                self.trade_log.append({
                    'action': 'buy',
                    'quantity': quantity,
                    'price': price,
                    'timestamp': timestamp
                })
            else:
                print("[回测模块] 资金不足，买入失败。")
        elif action == 'sell':
            if self.position >= quantity:
                revenue = quantity * price * (1 - fee_rate)
                self.current_capital += revenue
                self.position -= quantity
                profit = revenue - (quantity * trade.get('buy_price', price))
                self.trade_log.append({
                    'action': 'sell',
                    'quantity': quantity,
                    'price': price,
                    'profit': profit,
                    'timestamp': timestamp
                })
            else:
                print("[回测模块] 持仓不足，卖出失败。")

    def run_backtest(self, strategy_module, historical_data):
        """
        运行策略回测
        :param strategy_module: StrategyModule实例
        :param historical_data: DataFrame包含历史价格数据
        """
        for i in range(len(historical_data)):
            current_price = historical_data.iloc[i]['close']
            # 假设你的预测模型返回预测结果的DataFrame
            predictions = self.predict_price(historical_data.iloc[:i+1])
            
            if not predictions.empty:
                predicted_price = predictions.iloc[0]['predicted_price']
                lower_ci = predictions.iloc[0]['lower_ci']
                upper_ci = predictions.iloc[0]['upper_ci']
                
                action = strategy_module.execute(current_price, predicted_price, lower_ci, upper_ci, historical_data.iloc[:i+1])
                if action and action['status'] == 'approved':
                    self.dynamic_funds_management(action)
                    self.execute_trade(action)

    def predict_price(self, historical_data):
        """
        使用预测模型生成价格预测
        :param historical_data: DataFrame包含历史价格数据
        :return: DataFrame包含预测价格和置信区间
        """
        from model import hybrid_model  # 假设你的预测模型在model模块中

        try:
            # 确保historical_data的格式符合hybrid_model的要求
            historical_data.columns = historical_data.columns.str.lower()
            
            # 调用hybrid_model进行预测
            predictions = hybrid_model(
                config=self.config,
                data_path=None,  # 我们直接传入数据，不需要路径
                data=historical_data,  # 直接传递DataFrame
                lags=self.config["model_training"]["hybrid_model"].get("lags", 14),
                days_to_predict=1,  # 只预测一天
                ticker=self.config['ticker'],
                enable_self_learning=self.config["model_training"]["hybrid_model"].get("self_learning", True)
            )

            # 处理预测结果
            if not predictions.empty:
                return predictions[['timestamp', 'predicted_price', 'lower_ci', 'upper_ci']]
            else:
                print("预测结果为空，返回默认值")
                return pd.DataFrame([{'predicted_price': historical_data['close'].iloc[-1], 
                                    'lower_ci': historical_data['close'].iloc[-1] * 0.95, 
                                    'upper_ci': historical_data['close'].iloc[-1] * 1.05}])
        except Exception as e:
            print(f"预测错误: {e}")
            # 在错误情况下返回默认值
            return pd.DataFrame([{'predicted_price': historical_data['close'].iloc[-1], 
                                'lower_ci': historical_data['close'].iloc[-1] * 0.95, 
                                'upper_ci': historical_data['close'].iloc[-1] * 1.05}])
        
        
    def monte_carlo_simulation(self, strategy_class, param_ranges, num_simulations=100):
        """
        使用蒙特卡洛模拟优化策略参数。
        :param strategy_class: 策略类（如GridStrategy）。
        :param param_ranges: 参数范围（字典形式，例如 {'grid_size': (1, 10)}）。
        :param num_simulations: 模拟次数。
        """
        results = []

        for _ in range(num_simulations):
            params = {key: random.uniform(*value) for key, value in param_ranges.items()}
            strategy = strategy_class(**params)

            self.reset_backtest()
            strategy_module = StrategyModule([strategy], None, None, self.config)  # 需要修改为实际的交易和风险管理模块
            self.run_backtest(strategy_module, self.config['data_fetch']['historical_data'])
            
            cumulative_return = (self.current_capital / self.initial_capital) - 1
            sharpe_ratio = self.calculate_sharpe_ratio(self.get_returns())

            results.append({'params': params, 'cumulative_return': cumulative_return, 'sharpe_ratio': sharpe_ratio})

        results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        best_result = results[0]
        print(f"[蒙特卡洛模拟] 最佳参数: {best_result['params']}")
        print(f"[蒙特卡洛模拟] 夏普比率: {best_result['sharpe_ratio']:.2f}")
        return best_result


    def optimize_strategy(self, strategies, params_grid):
        """
        策略参数网格搜索
        :param strategies: 策略列表
        :param params_grid: 参数网格
        """
        best_params = {}
        best_performance = -np.inf

        for strategy, param_combination in product(strategies, params_grid):
            strategy.set_params(**param_combination)
            self.reset_backtest()
            self.run_strategy(strategy)
            performance = self.calculate_cumulative_return()

            if performance > best_performance:
                best_performance = performance
                best_params = param_combination

        print(f"[回测模块] 最佳参数组合: {best_params}")
        return best_params

    def evaluate_strategy_combinations(self, strategies):
        """
        评估策略组合的协同效果
        :param strategies: 策略列表
        """
        combination_results = {}
        for strategy_combination in product(strategies, repeat=2):
            if strategy_combination[0] == strategy_combination[1]:
                continue
            self.reset_backtest()
            for strategy in strategy_combination:
                self.run_strategy(strategy)
            performance = self.calculate_cumulative_return()
            combination_results[strategy_combination] = performance
        return combination_results

    def assess_risk_levels(self):
        """
        评估不同风险水平下的策略表现
        """
        risk_results = {}
        for risk_level in self.risk_levels:
            self.config['risk_management']['risk_tolerance'] = risk_level
            self.reset_backtest()
            self.run_strategy(self.config['backtest']['default_strategy'])
            performance = self.calculate_cumulative_return()
            risk_results[risk_level] = performance
        return risk_results

    def dynamic_funds_management(self, trade):
        """
        动态资金管理
        :param trade: 交易信号
        """
        risk_tolerance = self.config['risk_management']['risk_tolerance']
        trade_value = trade['quantity'] * trade['price']
        if trade_value > self.current_capital * risk_tolerance:
            trade['quantity'] = int((self.current_capital * risk_tolerance) / trade['price'])
            print(f"[回测模块] 动态调整交易量至: {trade['quantity']}")

    def calculate_performance(self):
        """
        计算表现指标
        """
        portfolio_value = self.current_capital + self.position * self.trade_log[-1]['price'] if self.trade_log else self.current_capital
        returns = [trade.get('profit', 0) for trade in self.trade_log if trade['action'] == 'sell']
        cumulative_return = (portfolio_value / self.initial_capital) - 1
        portfolio_values = [self.initial_capital + sum(returns[:i + 1]) for i in range(len(returns))]
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)

        print(f"累计收益率：{cumulative_return:.2%}")
        print(f"最大回撤：{max_drawdown:.2%}")
        print(f"夏普比率：{sharpe_ratio:.2f}")

        self.performance_metrics.append({
            'cumulative_return': cumulative_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        })

    @staticmethod
    def calculate_max_drawdown(portfolio_values):
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0):
        if np.std(returns) == 0:
            return 0
        return (np.mean(returns) - risk_free_rate) / np.std(returns)

    def visualize_results(self, historical_data):
        portfolio_values = [self.initial_capital]
        for trade in self.trade_log:
            if trade['action'] == 'sell':
                portfolio_values.append(portfolio_values[-1] + trade['profit'])
            else:
                portfolio_values.append(portfolio_values[-1])

        # 投资组合价值曲线
        plt.figure(figsize=(14, 7))
        plt.plot(historical_data.index, portfolio_values, label='Portfolio Value')
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Trades")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()

        # 收益分布图
        returns = self.get_returns()  # 从交易日志中提取收益
        plt.figure(figsize=(14, 7))
        plt.hist(returns, bins=30, alpha=0.75, label='Returns Distribution')
        plt.title("Returns Distribution")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        plt.show()

        # 风险分布图
        risks = [self.calculate_risk(trade) for trade in self.trade_log]
        plt.figure(figsize=(14, 7))
        plt.hist(risks, bins=30, alpha=0.75, color='orange', label='Risk Distribution')
        plt.title("Risk Distribution")
        plt.xlabel("Risk Exposure")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        plt.show()

    def analyze_strategy_performance(self):
        strategy_stats = {}
        for trade in self.trade_log:
            strategy = trade.get('strategy', 'unknown')
            profit = trade.get('profit', 0)
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'count': 0, 'total_profit': 0}
            strategy_stats[strategy]['count'] += 1
            strategy_stats[strategy]['total_profit'] += profit

        for strategy, stats in strategy_stats.items():
            print(f"策略 {strategy}：触发次数 {stats['count']}，总收益 {stats['total_profit']:.2f}")



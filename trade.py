from datetime import datetime, timedelta
import time
from riskMonitor import RiskManagementModule 
from ib_insync import IB, Stock, MarketOrder, LimitOrder

class RealTimeTradingModule:
    def __init__(self, trading_config, global_params, account_status, risk_management, ib=None):
        """
        初始化交易模块
        :param trading_config: 实时交易相关配置
        :param global_params: 全局参数
        :param account_status: 当前账户状态
        :param risk_management: 风险管理模块实例
        """

        if ib is None:
            self.ib = IB()
            self.ib.connect('127.0.0.1', 7497, clientId=1)
        else:
            self.ib = ib

        self.trading_config = trading_config
        self.global_params = global_params
        self.risk_management = risk_management

        self.orders = []  # 存储订单信息
        self.account_status = account_status
        self.max_retry_attempts = trading_config.get('max_retry_attempts', 3)
        self.retry_interval = trading_config.get('retry_interval', 5)
        self.fee_rate = global_params.get('fee_rate', 0.001)
        self.slippage_rate = global_params.get('slippage_rate', 0.001)

    def get_account_status(self):
        return self.account_status

    def update_account_status(self):
        account_values = self.ib.accountValues()
        for value in account_values:
            if value.tag == 'NetLiquidation':  # 净资产
                self.account_status['capital'] = float(value.value)
                print(f"[投资提示] 当前账户净资产：${self.account_status['capital']:.2f}")
            elif value.tag == 'TotalCashValue':  # 总现金
                print(f"[投资提示] 账户总现金：${float(value.value):.2f}")
            elif value.tag == 'GrossPositionValue':  # 总持仓价值
                self.account_status['position_value'] = float(value.value)
                print(f"[投资提示] 总持仓价值：${self.account_status['position_value']:.2f}")
        
        # 更新持仓信息，这里假设你有方法获取持仓详细信息
        positions = self.ib.positions()
        self.account_status['positions'] = {pos.contract.symbol: pos.position for pos in positions}
        print(f"[投资提示] 当前持仓：{', '.join([f'{symbol} x {qty}' for symbol, qty in self.account_status['positions'].items()])}")

        # 资产配置比例提示（示例）
        if self.account_status.get('position_value', 0) > 0:
            equity_percentage = (self.account_status['position_value'] / self.account_status['capital']) * 100
            print(f"[投资提示] 股票资产占比：{equity_percentage:.2f}%")

    def submit_order(self, action, quantity, price, ticker, order_type='market'):
        contract = Stock(ticker, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        
        if action.lower() == 'buy':
            action = 'BUY'
        elif action.lower() == 'sell':
            action = 'SELL'
        else:
            raise ValueError("无效的交易动作")

        if order_type.lower() == 'limit':
            order = LimitOrder(action, quantity, price)
        else:
            order = MarketOrder(action, quantity)

        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)  # 等待订单状态更新
        if trade.orderStatus.status == 'Filled':
            print(f"[交易模块] 订单已执行 - {trade.orderStatus.status}: {trade.orderStatus}")
            # 更新账户状态
            self.update_account_status(action, quantity, trade.orderStatus.avgFillPrice)
            return {'status': 'success', 'order_id': trade.order.orderId}
        else:
            print(f"[交易模块] 订单未执行 - {trade.orderStatus.status}")
            return {'status': 'failed', 'message': trade.orderStatus.status}

    def execute_trade(self, action_data, ticker):
        """
        执行交易
        :param action_data: 风险监测模块校验后的交易指令
        :param ticker: 股票代码
        """
        if not action_data or action_data.get('status') == 'blocked':
            print(f"[交易模块] 交易被阻止，原因：{action_data.get('reason', '无理由')}")
            return

        # 验证核心字段
        required_keys = {"action", "quantity", "price"}
        for key in required_keys:
            if key not in action_data or not action_data[key]:
                print(f"[交易模块] 无效交易指令，缺少字段 {key} 或值无效")
                return

        action = action_data['action']
        quantity = action_data['quantity']
        price = action_data['price']
        order_type = action_data.get('order_type', 'market')  # 默认市场订单

        retries = 0
        while retries < self.max_retry_attempts:
            try:
                # 提交订单
                result = self.submit_order(action=action, quantity=quantity, price=price, ticker=ticker, order_type=order_type)
                if result['status'] == 'success':
                    print(f"[交易模块] 交易完成：订单ID {result['order_id']} 状态成功。")
                    self.update_account_status()  # 更新账户状态
                    return
            except Exception as e:  # 捕获所有异常
                print(f"[交易模块] 订单提交失败：{e}")

            retries += 1
            print(f"[交易模块] 交易失败：尝试重试 {retries}/{self.max_retry_attempts}")
            time.sleep(self.retry_interval)

        print("[交易模块] 超过最大重试次数，交易失败。")

        # 即使交易失败，你也可能希望更新账户状态，因为市场可能会有变化
        self.update_account_status()
        print("[交易模块] 账户状态已更新，尽管交易失败。")

    def cancel_order(self, order_id):
        try:
            for order in self.ib.orders():
                if order.orderId == order_id:
                    self.ib.cancelOrder(order)
                    print(f"[交易模块] 已取消订单 ID: {order_id}")
                    return True
            print(f"[交易模块] 找不到订单 ID: {order_id}")
            return False
        except Exception as e:
            print(f"[交易模块] 取消订单时发生错误: {e}")
            return False

    def trade_summary(self):
        """
        返回交易总结信息
        """
        return {
            "Total Orders": len(self.orders),
            "Last Order": self.orders[-1] if self.orders else None,
            "Current Account Status": self.account_status
        }

    def validate_account_status(self):
        """
        校验账户状态
        """
        if self.account_status['capital'] < 0:
            raise ValueError("[错误] 账户资金异常！")
        if self.account_status['position'] < 0:
            raise ValueError("[错误] 持仓异常！")
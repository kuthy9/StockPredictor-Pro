import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from techlag_index import calculate_technical_indicators, create_lag_features
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import datetime
import pytz



# ====================== 模型训练部分 ======================
def train_lstm(X_train, y_train, input_shape, config):

    lstm_config = config['model_training']['lstm']


    """
    优化后的 LSTM 模型训练函数
    """
    print("开始训练优化后的 LSTM 模型...")
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_config['lstm_units'][0], return_sequences=True),
        Dropout(lstm_config['dropout_rate']),
        LSTM(lstm_config['lstm_units'][1], return_sequences=True),
        Dropout(lstm_config['dropout_rate']),
        LSTM(lstm_config['lstm_units'][2], return_sequences=False),
        Dropout(lstm_config['dropout_rate']),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 引入 Early Stopping
    early_stopping = EarlyStopping(
        monitor='loss',  # 监控训练集的损失
        patience=lstm_config['patience'],      # 如果连续 5 次迭代无显著改进，停止训练
        restore_best_weights=True  # 恢复最佳权重
    )

    # 开始训练
    model.fit(X_train, y_train, epochs=lstm_config['epochs'], 
            batch_size=lstm_config['batch_size'], verbose=1, 
            callbacks=[early_stopping])
    return model

# XGBoost 训练
def train_xgboost(X_train, y_train, config):

    print("开始训练 XGBoost 模型...")

    xgboost_config = config['model_training']['xgboost']

    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=xgboost_config['n_estimators'],
        max_depth=xgboost_config['max_depth'],
        learning_rate=xgboost_config['learning_rate']
    )
    # 确保输入特征 X_train 包括情绪特征
    model.fit(X_train, y_train)
    return model


# 新数据模型微调
def update_model(lstm_model, xgb_model, X_new, y_new, scaler, lags):
    print("开始自学习更新模型...")
    # 调整 LSTM 输入形状
    X_new_lstm = X_new.reshape((X_new.shape[0], lags + 1, 1))  # +1 包含情绪特征
    lstm_model.fit(X_new_lstm, y_new, epochs=5, 
                batch_size=32, verbose=0)
    # 调整 XGBoost 输入
    xgb_model.fit(X_new, y_new)
    print("LSTM 和 XGBoost 模型更新完成。")
    return lstm_model, xgb_model


# ====================== 绘图功能 ======================
def plot_predictions(df, hybrid_preds_actual, y_train, y_test_actual, 
                     scaler, future_timestamps, future_predictions, 
                     future_predictions_lower, future_predictions_upper, 
                     ticker):
    """
    修复后的绘图函数，确保 Hybrid Predictions 覆盖完整时间范围，同时增强分析功能
    """
    fig = go.Figure()

    # 添加实际价格
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['close'], 
        mode='lines', 
        name='Actual Prices',
        line=dict(color='blue')
    ))

    # 添加混合预测（完整范围）
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=hybrid_preds_actual.flatten(), 
        mode='lines', 
        name='Hybrid Predictions',
        line=dict(color='red', dash='dash')
    ))

    # 添加未来预测
    fig.add_trace(go.Scatter(
        x=future_timestamps, 
        y=future_predictions, 
        mode='lines', 
        name='Future Predictions',
        line=dict(color='green', dash='dot')
    ))

    # 添加预测区间上下界
    if len(future_predictions_lower) > 0 and len(future_predictions_upper) > 0:
        fig.add_trace(go.Scatter(
            x=future_timestamps,
            y=future_predictions_lower,
            mode='lines',
            name='Prediction Lower Bound',
            line=dict(color='gray', dash='dot', width=1),
            opacity=0.3  # 增加透明度
        ))
        fig.add_trace(go.Scatter(
            x=future_timestamps,
            y=future_predictions_upper,
            mode='lines',
            name='Prediction Upper Bound',
            line=dict(color='gray', dash='dot', width=1),
            opacity=0.3
        ))
    else:
        print("Prediction bounds are empty. Verify bound calculation logic.")

    # 填充预测区间
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_timestamps, future_timestamps[::-1]]),
        y=np.concatenate([future_predictions_upper, future_predictions_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',  # 半透明灰色填充
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Prediction Interval'
    ))

    # 标记高误差点（训练集）
    train_size = len(y_train)
    hybrid_train_preds_actual = hybrid_preds_actual[:train_size]
    train_high_error_indices = np.where(
        np.abs(y_train.flatten() - hybrid_train_preds_actual.flatten()) > 
        np.percentile(np.abs(y_train.flatten() - hybrid_train_preds_actual.flatten()), 90)
    )[0]
    train_high_error_timestamps = df['timestamp'].iloc[:train_size].iloc[train_high_error_indices]
    train_high_error_values = scaler.inverse_transform(y_train.flatten().reshape(-1, 1))[train_high_error_indices].flatten()
    fig.add_trace(go.Scatter(
        x=train_high_error_timestamps, y=train_high_error_values,
        mode='markers', name='Train High Error Points',
        marker=dict(color='red', size=8)
    ))

    # 标记高误差点（仅测试集范围）
    test_size = len(y_test_actual)
    hybrid_test_preds_actual = hybrid_preds_actual[-test_size:]  # 提取测试集范围的预测
    high_error_indices = np.where(
        np.abs(y_test_actual.flatten() - hybrid_test_preds_actual.flatten()) > 
        np.percentile(np.abs(y_test_actual.flatten() - hybrid_test_preds_actual.flatten()), 90)
    )[0]
    high_error_timestamps = df['timestamp'].iloc[-test_size:].iloc[high_error_indices]
    high_error_values = y_test_actual.flatten()[high_error_indices]
    fig.add_trace(go.Scatter(
        x=high_error_timestamps, y=high_error_values,
        mode='markers', name='High Error Points',
        marker=dict(color='orange', size=8)
    ))

    # 未来高误差点
    future_high_error_indices = np.where(
        np.abs(future_predictions - np.mean(future_predictions)) > 
        np.percentile(np.abs(future_predictions - np.mean(future_predictions)), 90)
    )[0]
    future_high_error_timestamps = [future_timestamps[i] for i in future_high_error_indices]
    future_high_error_values = future_predictions[future_high_error_indices]
    fig.add_trace(go.Scatter(
        x=future_high_error_timestamps, y=future_high_error_values,
        mode='markers', name='Future High Error Points',
        marker=dict(color='purple', size=8)
    ))


    # 图表布局
    fig.update_layout(
        title=f"Hybrid Model Predictions for {ticker}",  # 添加标题
        xaxis=dict(
            title="Date",
            tickformat="%b-%Y",  # 显示月份和年份
            rangeslider=dict(visible=True),  # 添加缩放功能
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        yaxis=dict(
            title="Price"
        )
    )
    fig.show()



# ===================== 自学习机制 ======================
def self_learning(config, lstm_model, xgb_model, X_train, y_train, lstm_preds, xgb_preds, scaler, lags, iterations=5):
    """
    自学习机制：动态调整高误差样本的权重，并迭代优化 LSTM 和 XGBoost 模型。
    """
    self_learning_config = config['model_training']['self_learning']

    # 从配置文件加载参数
    iterations = self_learning_config['iterations']
    error_threshold_percentile = self_learning_config['error_threshold_percentile']
    sample_weight_factor = self_learning_config['sample_weight_factor']

    for iteration in range(iterations):
        print(f"第 {iteration + 1} 次自学习...")

        # 计算混合预测误差
        hybrid_preds = 0.5 * lstm_preds + 0.5 * xgb_preds
        errors = np.abs(y_train - hybrid_preds)

        # 动态计算高误差阈值
        high_error_threshold = np.percentile(errors, error_threshold_percentile)
        high_error_indices = np.where(errors > high_error_threshold)[0]

        if len(high_error_indices) == 0:
            print("未找到高误差样本，自学习提前结束。")
            break

        # 获取高误差样本
        X_high_error = X_train[high_error_indices]
        y_high_error = y_train[high_error_indices]

        # 动态调整样本权重（基于误差大小）
        sample_weights = errors[high_error_indices] / errors[high_error_indices].sum()
        extreme_indices = y_high_error > np.percentile(y_high_error, 90)
        sample_weights[extreme_indices] *= sample_weight_factor  # 提升极端值的权重

        # 更新 LSTM 和 XGBoost 模型
        try:
            lstm_model.fit(
                X_high_error.reshape(len(X_high_error), lags, 1),
                y_high_error,
                sample_weight=sample_weights,
                epochs=5,
                batch_size=16,
                verbose=0
            )
            xgb_model.fit(X_high_error, y_high_error, sample_weight=sample_weights)
        except Exception as e:
            print(f"模型更新失败：{e}")
            break

        # 重新计算预测
        lstm_preds = lstm_model.predict(X_train.reshape(len(X_train), lags, 1)).flatten()
        xgb_preds = xgb_model.predict(X_train)

        # 动态调整 alpha
        lstm_error = np.mean(np.abs(y_train - lstm_preds))
        xgb_error = np.mean(np.abs(y_train - xgb_preds))
        total_error = lstm_error + xgb_error
        alpha = (1 - (lstm_error / total_error)) * 0.5 + 0.5  # 动态调整权重

        # 输出调试信息
        print(f"第 {iteration + 1} 次自学习完成，高误差样本数量：{len(high_error_indices)}")
        print(f"动态调整后的 Alpha 值：{alpha:.4f}")
        print(f"LSTM 误差更新：{lstm_error:.4f}")
        print(f"XGBoost 误差更新：{xgb_error:.4f}")

        # 提前终止机制：高误差样本过少
        if len(high_error_indices) < len(X_train) * 0.01:
            print("高误差样本过少，自学习提前结束。")
            break

    return lstm_model, xgb_model


# ====================== 主模型函数 ======================
def hybrid_model(config, data_path, lags=None, days_to_predict=None, ticker="Unknown", enable_self_learning=True):
    hybrid_config = config['model_training']['hybrid_model']
    
    # 获取清洗后的数据路径
    data_path = config['data_clean']['data_path']['cleaned_data_path']  # 从清洗模块读取数据路径
    data_file_path = f"./data/cleaned/cleaned_{ticker}_5y.parquet"

    # 提取配置参数
    lags = hybrid_config['lags']
    days_to_predict = hybrid_config['future_prediction_days']
    simulations = hybrid_config['simulations']
    percentage = hybrid_config['percentage']
    alpha_min = hybrid_config['alpha_min']
    alpha_max = hybrid_config['alpha_max']
    expand_factor = hybrid_config['expand_factor']
    enable_self_learning = hybrid_config['self_learning']

    # 加载清洗后的数据
    print(f"正在加载清洗后的数据文件：{data_file_path}")
    df = pd.read_parquet(data_file_path)
    df.columns = df.columns.str.lower()

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise ValueError("时间戳列不是有效的日期时间格式，请检查数据清洗模块。")
    print(f"清洗后数据时间范围：{df['timestamp'].min()} 至 {df['timestamp'].max()}")


    # 计算技术指标
    print("计算技术指标...")
    df = calculate_technical_indicators(
        df,indicators=['sma_5', 'ema_5', 'rsi', 'macd', 'bollinger', 
                    'atr', 'vwap', 'volume_oscillator',
                    'momentum', 'stochastic', 'cci', 'roc', 
                    'williams_r', 'adx']
    )

    # 生成滞后特征
    print("生成滞后特征...")
    df = create_lag_features(df, indicators=['close', 'rsi', 'macd_histogram', 'cci', 'roc', 'adx'], lags=lags)

    # 动态调整后检查数据完整性
    for column in df.columns:
        if df[column].isnull().any():
            print(f"特征 {column} 存在 NaN 值，调整后数据存在问题。")
        if np.isinf(df[column].values).any():
            print(f"特征 {column} 存在无效值 (inf)，调整后数据存在问题。")

    # 如果检查失败，抛出异常
    if df.isnull().any().any():
        raise ValueError("动态调整后的数据包含 NaN 或缺失值，请检查调整逻辑。")

    # 标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['close_scaled'] = scaler.fit_transform(df[['close']])

    X = df[[f'lag_close_{i}' for i in range(1, lags + 1)]].values
    y = df['close_scaled'].values

    train_size = int(len(X) * percentage)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")
    print(f"y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")

    X_train_lstm = X_train.reshape((X_train.shape[0], lags, 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], lags, 1))

    # 检查 LSTM 训练数据是否存在问题
    if np.any(np.isnan(X_train_lstm)) or np.any(np.isnan(y_train)):
        raise ValueError("LSTM 训练数据包含 NaN 或无效值，请检查数据生成和转换逻辑。")

    # 训练 LSTM 和 XGBoost 模型
    lstm_model = train_lstm(X_train_lstm, y_train, input_shape=(lags, 1), config=config)


    # 训练 XGBoost 模型
    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
        raise ValueError("XGBoost 训练数据包含 NaN 或无效值，请检查数据生成逻辑。")
    xgb_model = train_xgboost(X_train, y_train, config=config)

    # 预测训练集
    lstm_train_preds = lstm_model.predict(X_train_lstm).flatten()
    xgb_train_preds = xgb_model.predict(X_train)

    # 启动自学习
    if enable_self_learning:
        print("启动自学习机制...")
        lstm_model, xgb_model = self_learning(
            config, lstm_model, xgb_model, X_train, y_train,
            lstm_train_preds, xgb_train_preds, scaler, lags
        )

    # 动态调整 alpha
    alpha = 0.7

    # 混合预测训练集
    train_hybrid_preds = alpha * lstm_train_preds + (1 - alpha) * xgb_train_preds

    # 预测测试集
    lstm_test_preds = lstm_model.predict(X_test_lstm).flatten()
    xgb_test_preds = xgb_model.predict(X_test)
    test_hybrid_preds = alpha * lstm_test_preds + (1 - alpha) * xgb_test_preds

    # 合并预测结果
    hybrid_preds = np.concatenate([train_hybrid_preds, test_hybrid_preds])
    hybrid_preds_actual = scaler.inverse_transform(hybrid_preds.reshape(-1, 1))

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    print(f"混合模型 MSE: {mean_squared_error(y_test_actual, hybrid_preds_actual[-len(y_test_actual):]):.4f}")

    # 未来预测
    future_predictions = []
    future_timestamps = []
    future_predictions_ensemble = []
    last_date = df['timestamp'].iloc[-1]
    latest_features = X_test[-1].reshape(1, -1)

    for _ in range(simulations):  # 多次模拟
        future_predictions_simulation = []
        features = latest_features.copy()

        for i in range(1, days_to_predict + 1):
            alpha_future = np.clip(alpha * (1 - i / days_to_predict), alpha_min, alpha_max)
            lstm_pred = lstm_model.predict(features[:, :lags].reshape(1, lags, 1))
            xgb_pred = xgb_model.predict(features)

            # 动态权重预测
            future_pred_scaled = alpha_future * lstm_pred.flatten() + (1 - alpha_future) * xgb_pred
            future_pred = scaler.inverse_transform(future_pred_scaled.reshape(-1, 1))[0, 0]

            # 异常值修正
            mean_price = df['close'].mean()
            std_price = df['close'].std()
            upper_bound = mean_price + 3 * std_price
            lower_bound = max(mean_price - 3 * std_price, df['close'].min() * 0.8)
            
            range_multiplier = 1 + 0.01 * (i / days_to_predict)
            if future_pred > upper_bound:
                future_pred = upper_bound * range_multiplier
            elif future_pred < lower_bound:
                future_pred = lower_bound / range_multiplier

            # 更新滞后特征
            sma = np.mean(features[0, -lags:])
            features = np.roll(features, -1)
            features[:, -1] = 0.8 * future_pred + 0.2 * sma

            future_predictions_simulation.append(future_pred)

        future_predictions_ensemble.append(future_predictions_simulation)

    # 将多次模拟结果平均
    future_predictions = np.mean(future_predictions_ensemble, axis=0)

    # 计算上下界，并适当扩展
    expand_factor = 1.2
    future_predictions_lower = np.percentile(future_predictions_ensemble, 10, axis=0) * (2 - expand_factor)
    future_predictions_upper = np.percentile(future_predictions_ensemble, 90, axis=0) * expand_factor

    # 动态扩展上下界
    future_predictions_lower = np.maximum(future_predictions_lower, future_predictions * 0.9)
    future_predictions_upper = np.minimum(future_predictions_upper, future_predictions * 1.1)
    
    # 生成未来时间戳
    current_date = datetime.datetime.now(pytz.timezone('America/Toronto')).date()
    # 生成包含当天的预测
    future_timestamps = [pd.Timestamp(current_date, tz='America/Toronto') + pd.Timedelta(days=i) for i in range(days_to_predict)]


    print("Current Date:", current_date)
    print("First future timestamp:", future_timestamps[0])

    # 绘图
    plot_predictions(
        df=df,
        hybrid_preds_actual=hybrid_preds_actual,
        y_train=y_train,
        scaler=scaler,
        y_test_actual=y_test_actual,
        future_timestamps=future_timestamps,
        future_predictions=future_predictions,
        future_predictions_lower=future_predictions_lower,
        future_predictions_upper=future_predictions_upper,
        ticker=ticker
    )

    # 获取预测结果存储路径
    output_file = config['model_training']['data_path']['prediction_path'].format(ticker=ticker)

    try:
        predictions = pd.DataFrame({
            'timestamp': future_timestamps,
            'predicted_price': future_predictions,
            'lower_ci': future_predictions_lower,
            'upper_ci': future_predictions_upper
        }).set_index('timestamp')

        print("预测结果的列名:", predictions.columns)
        print("预测结果的索引类型:", type(predictions.index))
        if isinstance(predictions.index, pd.DatetimeIndex):
            print("预测结果的时区:", predictions.index.tz)


        # 保存预测结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_price': future_predictions,
        'lower_ci': future_predictions_lower,
        'upper_ci': future_predictions_upper
        })

        predictions.to_parquet(output_file, engine='pyarrow')

        # 立即读取并验证
        verified_data = pd.read_parquet(output_file)
        print("Verified timestamps:", verified_data.index.min(), "to", verified_data.index.max())
        # 打印调试信息
        print("保存后检查索引类型:", type(verified_data.index))
        print("保存后检查时区:", verified_data.index.tz)
        print("预测数据 DataFrame 列:", pd.DataFrame({
            'timestamp': future_timestamps,
            'predicted_price': future_predictions,
            'lower_ci': future_predictions_lower,
            'upper_ci': future_predictions_upper
        }).columns)


        print(f"预测结果已保存到 {output_file}")

    except KeyError as e:
        print(f"KeyError: {e} - 'timestamp' 列未找到")
    except AttributeError as e:
        print(f"AttributeError: {e} - 可能尝试在非DataFrame对象上访问列")
    except Exception as e:
        print(f"其他错误: {e}")

    # 可以选择返回 predictions DataFrame 或其他值
    return predictions

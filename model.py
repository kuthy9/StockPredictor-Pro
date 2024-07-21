import pandas as pd
import numpy as np
import os
import logging
import yaml
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input, Conv1D, LayerNormalization, Attention
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from scikeras.wrappers import KerasRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import xgboost as xgb

# 加载配置文件
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# 设置日志记录
log_dir = config['log_dir']
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'train_model.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = config['data_dir']
processed_data_dir = os.path.join(data_dir, 'processed')

# 生成数据集函数，将时间序列数据转换为监督学习数据
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# 定义LSTM模型
def create_lstm_model(look_back, input_shape, optimizer='adam', dropout_rate=0.2, units=50, l2_reg=0.01):
    input_layer = Input(shape=(look_back, input_shape))
    
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)  # 添加1D卷积层
    x = LSTM(units, return_sequences=True, kernel_regularizer=l2(l2_reg))(x)  # 添加LSTM层
    x = LayerNormalization()(x)  # 添加层归一化
    attention_out = Attention()([x, x])  # 添加自注意力机制层
    x = LSTM(units, kernel_regularizer=l2(l2_reg))(attention_out)  # 添加第二个LSTM层
    x = Dropout(dropout_rate)(x)  # 添加Dropout层
    output = Dense(1)(x)  # 添加输出层

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)  # 编译模型
    return model

# 主训练函数
def train_model():
    # 读取处理后的数据
    processed_file_path = os.path.join(processed_data_dir, 'processed_stock_data.csv')
    data = pd.read_csv(processed_file_path)

    # 确保所有需要的列都存在
    required_columns = ['CROCI', 'EBITDA', 'EPS_Growth', 'P/E', 'P/FCF', 'P/B', 'P/S', 'Gross_Margin', 
                        'Piotroski_F_Score', 'ROA', 'Sales_QQ', '52_week_high_percent', 'rsi', 'rsi_slope', 
                        'macd', 'macdsignal', 'macdhist', 'willr', 'mfi', 'Sentiment']

    for col in required_columns:
        if (col not in data.columns):
            raise ValueError(f"Data must contain '{col}' column.")

    # 填充缺失值
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    # 检查是否仍有 NaN 值
    if data.isnull().values.any():
        raise ValueError("Data still contains NaN values after filling.")

    # 将数据转换为 numpy 数组，排除非数值列
    dataset = data[required_columns].select_dtypes(include=[np.number]).values.astype('float32')

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 分割数据集为训练和测试
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[:train_size], dataset[train_size:]

    look_back = 60  # 将look_back设定为60个交易日
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # 检查X_train和X_test的维度
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("The dataset is too small for the given look_back period. Please use a smaller look_back period or a larger dataset.")

    # 调整数据维度为LSTM输入格式 [样本数, 时间步, 特征数]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # 使用GridSearchCV进行超参数调优
    param_grid = {
        'model__optimizer': ['adam', 'rmsprop'],
        'model__units': [50, 100],
        'model__l2_reg': [0.01, 0.001],
        'epochs': [5],  # 设置epoch的值
        'batch_size': [1]
    }

    lstm_model = KerasRegressor(model=create_lstm_model, look_back=look_back, input_shape=X_train.shape[2], verbose=0, dropout_rate=0.2)
    tscv = TimeSeriesSplit(n_splits=5)  # 使用时间序列交叉验证
    grid = GridSearchCV(estimator=lstm_model, param_grid=param_grid, cv=tscv)
    
    # 修改fit方法传递单个输入
    grid_result = grid.fit(X_train, Y_train)

    # 打印最佳参数
    logging.info(f"Best params: {grid_result.best_params_}")

    # 使用最佳参数训练最终模型
    best_params = grid_result.best_params_
    model_params = {key.split('__')[1]: value for key, value in best_params.items() if key.startswith('model__')}
    fit_params = {key: value for key, value in best_params.items() if not key.startswith('model__')}
    lstm_regressor = KerasRegressor(model=create_lstm_model, look_back=look_back, input_shape=X_train.shape[2], **model_params, dropout_rate=0.2)
    lstm_regressor.fit(X_train, Y_train, **fit_params)

    # LSTM预测
    trainPredict_LSTM = lstm_regressor.predict(X_train)
    testPredict_LSTM = lstm_regressor.predict(X_test)

    # 重塑数据为2D格式
    X_train_2d = X_train.reshape((X_train.shape[0], -1))
    X_test_2d = X_test.reshape((X_test.shape[0], -1))

    # 将LSTM预测结果作为新的特征加入到2D特征中
    X_train_combined = np.hstack((X_train_2d, trainPredict_LSTM.reshape(-1, 1)))
    X_test_combined = np.hstack((X_test_2d, testPredict_LSTM.reshape(-1, 1)))

    # 创建LightGBM和随机森林模型
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,  # 增加树的数量
        learning_rate=0.05,  # 调整学习率
        max_depth=15  # 增加最大深度
    )
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10)

    # 使用Stacking方法集成模型
    stacking_model = StackingRegressor(
        estimators=[
            ('xgb', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)),
            ('lgb', lgb_model),
            ('rf', rf_model)
        ],
        final_estimator=RandomForestRegressor(n_estimators=50)
    )

    # 训练Stacking模型
    stacking_model.fit(X_train_combined, Y_train)

    # 预测
    trainPredict = stacking_model.predict(X_train_combined)
    testPredict = stacking_model.predict(X_test_combined)

    # 反归一化预测值
    trainPredict = scaler.inverse_transform(np.concatenate((trainPredict.reshape(-1, 1), np.zeros((trainPredict.shape[0], dataset.shape[1] - 1))), axis=1))[:, 0]
    Y_train = scaler.inverse_transform(np.concatenate((Y_train.reshape(-1, 1), np.zeros((Y_train.reshape(-1, 1).shape[0], dataset.shape[1] - 1))), axis=1))[:, 0]
    testPredict = scaler.inverse_transform(np.concatenate((testPredict.reshape(-1, 1), np.zeros((testPredict.shape[0], dataset.shape[1] - 1))), axis=1))[:, 0]
    Y_test = scaler.inverse_transform(np.concatenate((Y_test.reshape(-1, 1), np.zeros((Y_test.reshape(-1, 1).shape[0], dataset.shape[1] - 1))), axis=1))[:, 0]

    # 预测未来的股票价格
    future_steps = 30
    future_input = dataset[-look_back:]
    future_input_reshaped = np.reshape(future_input, (1, look_back, X_train.shape[2]))
    future_predictions = []

    for _ in range(future_steps):
        future_pred_lstm = lstm_regressor.predict(future_input_reshaped)
        print(f"future_pred_lstm shape: {future_pred_lstm.shape}")  # 调试信息
        future_predictions.append(future_pred_lstm[0])
        future_pred_lstm_reshaped = np.reshape(future_pred_lstm, (1, 1, X_train.shape[2]))
        future_input_reshaped = np.concatenate([future_input_reshaped[:, 1:, :], future_pred_lstm_reshaped], axis=1)

    # 反归一化未来预测值
    future_predictions = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((future_steps, dataset.shape[1] - 1))), axis=1))[:, 0]
    logging.info("Future predictions: %s", future_predictions)

     # 计算误差
    trainScore = np.sqrt(np.mean((trainPredict - Y_train) ** 2))
    testScore = np.sqrt(np.mean((testPredict - Y_test) ** 2))
    logging.info(f'Train Score: {trainScore} RMSE')
    logging.info(f'Test Score: {testScore} RMSE')

if __name__ == "__main__":
    train_model()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.losses import Huber

# 读取CSV文件
data = pd.read_csv('C:/Users/dell/Desktop/pythonProject4/data/heart.csv')

# 查看数据
print(data.head())
#  将时间戳列转换为整数
data['Time (s)'] = data['Time (s)'].astype(int)
# 选择特征和标签
features = data[['Acceleration on chest', 'Acceleration on stomach', 'Acceleration on lower stomach']]
labels = data['Standard ECG signal']

# 标记极值点（大于 0.1 或小于 -0.1）
data['is_outlier'] = ((data['Standard ECG signal'] > 0.1) | (data['Standard ECG signal'] < -0.1)).astype(int)

# 将数据分为训练集和测试集，分层抽样保证极值点的比例
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=data['is_outlier']
)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 标准化标签
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# 将数据重塑为LSTM需要的形状 [samples, time steps, features]
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
# 根据极值点设置样本权重
sample_weights = np.where(
    (y_train.values > 0.1) | (y_train.values < -0.1),
    1.5,  # 极值点权重设为 1.5
    1.0  # 其他点权重设为 1.0
)
# 构建LSTM模型
model = Sequential()
model.add(Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))  # 输入层
model.add(LSTM(128, activation='relu', return_sequences=True))  # 第一层LSTM，带有返回序列
model.add(Dropout(0.2))  # Dropout层，防止过拟合

model.add(LSTM(64, activation='relu', return_sequences=True))  # 第二层LSTM
model.add(Dropout(0.2))  # 再次添加Dropout

model.add(LSTM(32, activation='relu'))  # 第三层LSTM
model.add(Dense(64, activation='relu'))  # 增加Dense层
model.add(Dense(1))  # 输出层

# 编译模型
model.compile(optimizer='adam', loss=Huber(delta=0.1))

# 使用 EarlyStopping 回调监控验证损失，并在没有改善时提前停止训练
early_stopping = EarlyStopping(
    monitor='val_loss',    # 监控验证损失
    patience=500,           # 如果验证损失在 100个 epoch 内没有改善，则停止训练
    restore_best_weights=True  # 恢复到具有最小验证损失的权重
)

# 训练模型并保存损失历史
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=3000,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_scaled),
    sample_weight=sample_weights,  # 将权重应用到训练中
    verbose=1,
    callbacks=[early_stopping]
)

# 评估模型
loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f'Test Loss: {loss}')

# 进行预测
y_pred_scaled = model.predict(X_test_scaled)

# 反标准化预测结果和真实值
y_pred_inverse = y_scaler.inverse_transform(y_pred_scaled)
y_test_inverse = y_scaler.inverse_transform(y_test_scaled)

# 监控预测值的最大最小值
y_pred_max = np.max(y_pred_inverse)
y_pred_min = np.min(y_pred_inverse)
print(f'Predicted Max Value: {y_pred_max}')
print(f'Predicted Min Value: {y_pred_min}')

# 筛选出极值点
is_outlier = data['is_outlier'][y_test.index] == 1
y_test_outliers = y_test_inverse[is_outlier]
y_pred_outliers = y_pred_inverse[is_outlier]

# 计算极值点上的均方误差
mse_outliers = mean_squared_error(y_test_outliers, y_pred_outliers)
print(f'Mean Squared Error on Outliers: {mse_outliers}')

# 绘制真实值和预测值的对比图
plt.figure(figsize=(12, 6))
plt.plot(y_test_outliers, label='True Outlier Values')
plt.plot(y_pred_outliers, label='Predicted Outlier Values', linestyle='--')
plt.xlabel('Outlier Sample Index')
plt.ylabel('Heart Rate')
plt.title('True vs Predicted Values on Outliers')
plt.legend()
plt.show()

# 可视化训练和验证损失
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()
# 输出最小验证损失
min_val_loss = min(history.history['val_loss'])
print(f'Minimum Validation Loss: {min_val_loss}')
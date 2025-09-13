import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

# 读取数据
data = pd.read_csv('C:/Users/dell/Desktop/pythonProject4/data/walk.csv')

# 选择特征和标签
features = data[['Acceleration on chest', 'Acceleration on stomach', 'Acceleration on lower stomach']]
labels = data['Standard ECG signal']

# 按顺序划分前 90% 为训练集，后 10% 为测试集
split_index = int(len(data) * 0.9)
X_train, X_test = features[:split_index], features[split_index:]
y_train, y_test = labels[:split_index], labels[split_index:]

# 标准化特征和标签
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# 设置时间步长
time_steps = 20

# 将数据重塑为 LSTM 需要的形状 [samples, time steps, features]
def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train_scaled, y_train_scaled = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_test_scaled, y_test_scaled = create_sequences(X_test_scaled, y_test_scaled, time_steps)

# 构建 LSTM 模型
model = Sequential([
    Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    LSTM(128, activation='tanh', return_sequences=True),
    Dropout(0.3),
    LSTM(64, activation='tanh'),
    Dropout(0.3),
    Dense(1)
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.0002), loss='mean_squared_logarithmic_error')

# 设置回调函数：早停和学习率衰减
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

# 训练模型
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 预测并反标准化
y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test = y_scaler.inverse_transform(y_test_scaled)

# 计算均方误差
mse = np.mean((y_pred - y_test) ** 2)
print("Mean Squared Error:", mse)
# 检查 y_test 和 y_pred 的最小值和最大值
print(f"True Values Min = {y_test.min()}, Max = {y_test.max()}")
print(f"Predicted Values Min = {y_pred.min()}, Max = {y_pred.max()}")

# 确保 y_test 和 y_pred 形状一致
print("Shape of y_test:", y_test.shape)
print("Shape of y_pred:", y_pred.shape)


# 绘制预测值与真实值的对比图
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Values', color='b')
plt.plot(y_pred, label='Predicted Values', color='y', linestyle='--')
plt.title("Prediction vs True Heart Rate Signal")
plt.xlabel("Sample Index")
plt.ylabel("Heart Rate Signal")
plt.legend()
plt.show()

# 绘制仅预测值的图
plt.figure(figsize=(12, 6))
plt.plot(y_pred, color='y', linestyle='--', label='Predicted Values')
plt.title("Predicted Heart Rate Signal")
plt.xlabel("Sample Index")
plt.ylabel("Heart Rate Signal")
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 读取数据
data = pd.read_csv('C:/Users/dell/Desktop/pythonProject4/data/walk.csv')  # 请替换为你的文件路径

# 选择特征和标签
features = data[['Acceleration on chest', 'Acceleration on stomach', 'Acceleration on lower stomach']]
labels = data['Standard ECG signal']
time_column = data['Time (s)']  # 使用 'Time (s)' 作为时间列

# 筛选标签值大于 0.4 或小于 -0.4 的数据点
threshold_positive = 0.4
threshold_negative = -0.4
condition = (labels > threshold_positive) | (labels < threshold_negative)
filtered_features = features[condition]
filtered_labels = labels[condition]
filtered_time_column = time_column[condition]

# 重置索引
filtered_features.reset_index(drop=True, inplace=True)
filtered_labels.reset_index(drop=True, inplace=True)
filtered_time_column.reset_index(drop=True, inplace=True)

# 按顺序划分前 90% 为训练集，后 10% 为测试集
split_index = int(len(filtered_features) * 0.9)
X_train, X_test = filtered_features[:split_index], filtered_features[split_index:]
y_train, y_test = filtered_labels[:split_index], filtered_labels[split_index:]
time_train, time_test = filtered_time_column[:split_index], filtered_time_column[split_index:]

# 标准化特征和标签
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# 设置时间步长
time_steps = 7

# 创建序列数据
def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

if len(X_train_scaled) > time_steps:
    X_train_scaled_seq, y_train_scaled_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
else:
    print("训练数据不足以创建序列，请减少 time_steps 或获取更多数据。")
    X_train_scaled_seq, y_train_scaled_seq = np.array([]), np.array([])

if len(X_test_scaled) > time_steps:
    X_test_scaled_seq, y_test_scaled_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)
else:
    print("测试数据不足以创建序列，请减少 time_steps 或获取更多数据。")
    X_test_scaled_seq, y_test_scaled_seq = np.array([]), np.array([])

# 检查是否有足够的数据进行训练
if X_train_scaled_seq.size == 0 or X_test_scaled_seq.size == 0:
    print("数据不足，无法训练模型。")
else:
    # 构建 LSTM 模型
    model = Sequential([
        Input(shape=(X_train_scaled_seq.shape[1], X_train_scaled_seq.shape[2])),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.0004), loss='mean_absolute_error')

    # 设置回调函数：早停和学习率衰减
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # 训练模型
    history = model.fit(
        X_train_scaled_seq, y_train_scaled_seq,
        epochs=1000,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # 预测并反标准化
    y_pred_scaled = model.predict(X_test_scaled_seq)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_test_actual = y_scaler.inverse_transform(y_test_scaled_seq).flatten()

    # 保存真实值和预测值到 CSV 文件
    results_df = pd.DataFrame({
        'True Values': y_test_actual,
        'Predicted Values': y_pred,
        'Time (s)': time_test[time_steps:].values  # 确保时间列与预测值长度一致
    })
    results_df.to_csv('C:/Users/dell/Desktop/pythonProject4/data/jog results.csv', index=False)
    print("预测结果已保存至 jog results.csv 文件")

    # 计算均方误差
    mse = np.mean((y_pred - y_test_actual) ** 2)
    print("Mean Squared Error:", mse)

    # 绘制预测值与真实值的对比图
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='True Values', color='b')
    plt.plot(y_pred, label='Predicted Values', color='y', linestyle='--')
    plt.title("Prediction vs True Heart Rate Signal (Filtered Data)")
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
    # 在图上添加时间差的注释
    plt.text(time_test.iloc[-1], max(y_test_actual.max(), y_pred.max()) * 0.9,
             f'Time Difference: {time_diff:.2f} s',
             verticalalignment='top', horizontalalignment='right', color='red', fontsize=8)

    plt.show()
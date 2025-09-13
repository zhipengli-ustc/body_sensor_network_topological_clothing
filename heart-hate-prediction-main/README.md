# heart-hate-prediction
在该心率预测的代码中，model分为all pred是预测六万个数据点的模型，由于数据噪声较大所以效果不好。jog run walk是用三个加速度计作为特征输入的模型，分别预测了
jog run walk状态下的心率，为了保证准确率仅预测大于0.4小于-0.4的极值点，time_diff会输出时间差，方便计算心率。single input模型用于单加速度计作为特征输入，来预测心率。
wave pred保留了大于0.1和小于-0.1的点，可以预测出心率曲线一次心跳中的的大波形和小波形。

在心率预测的过程中使用了LSTM模型，以三个加速度计作为特征输入，来预测心率变化。
首先尝试了all pred--去预测六万个数据点的以及wave pred可以预测出心率曲线一次心跳中的的大波形和小波形。但由于数据噪声较大都没有取得太好的预测效果，所以转入了极值点预测。


关于结果：
jog result，run result，walk result分别保存了相关状态的预测结果图和预测值csv文件
例如 图片名称jog 就是jog模型的结果，包含真实值和预测值的对比。jog pred就是仅包含预测值的可视化图像。
例如 图片名称single stomach-jog 就是single input模型用stomach加速计单特征预测jog时心率，真实值和预测值的对比图；single stomach-jog pred就是仅包含预测值的可视化图像。
例如 csv文件single stomach-jog就是stomach加速计单特征预测jog时保存预测值的csv文件


关于结论：考虑三个加速度的信号作为特征输入，walk状态下，预测结果和真实值较为吻合，jog状态下稍差，run状态下噪声较大，拟合情况较差
考虑单个加速度计作为特征输入，分别用chest 、stomach 、lower stomach作为单特征输入预测心率曲线，从结果分析出stomach作为单特征预测心率曲线效果最好，lower stomach作为单特征输入预测效果最差，甚至导致预测中负值被平滑的结果
stomach单特征预测run时要比，三特征输入预测run状态的心率结果准确率更高。预测walk是准确率相近，预测jog时准确率稍差

使用说明：替换文件路径即可

新增说明：针对于三个加速度计的信号作为输入预测jog run walk状态下的心率，进行了真实值和预测值对比的绘图，横坐标是时间，纵坐标心率的计算公式是|60/相邻极大值点的时间差值|。用同样的方法对chest加速计作为单独输入
预测jog run walk状态下的心率的预测结果进行了绘图。通局部峰值提取 (find_peaks) 提取指定列中的局部极大值索引，可以通过调节 height 参数筛除噪声较大的小峰值。绘图结果和绘图代码保存在heart_rate_results文件中。human fliter walk结果是
手动筛选了walk.result.csv中的极值点并绘图的结果，用于和自动提取极值点的结果做对比
https://github.com/YUZHEZHONG/heart-hate-prediction

import gc
import numpy as np
import pandas as pd
import pickle
from flaml import AutoML
import flaml

# 加载训练数据
train_data = np.load(r"H:\EXP2\Data\train.npy")

# 定义列名：前8个特征分别为曲率、坡向、坡度、高程、窨井密度、湿水指数、土地利用类型、降雨量，
# 接着是第9到第68列（共60列）表示不同时间的降雨相关特征 Rain_i，
# 最后一列为目标变量 Y
cols = ["Curvature", "Aspect", "Slope", "Elevation", "Manhole density", "TWI", "Land use", "Rainfall"]
for i in range(len(cols), 68):
    cols.append(f"Rain_{i}")
cols.append("Y")

# 将numpy数组转换为DataFrame，并指定列名
df = pd.DataFrame(train_data, columns=cols)


def train_method(m):
    """
    使用FLAML框架自动训练单个模型并保存结果。

    参数:
        m (list): 包含一个字符串元素的列表，代表要使用的机器学习算法名称，例如 ['lgbm']。

    返回值:
        无返回值。将训练好的AutoML对象序列化存储为.pkl文件。
    """
    # 设置AutoML配置参数
    automl_settings = {
        "metric": 'rmse',  # 评估指标为均方根误差
        "task": 'regression',  # 任务类型为回归
        "log_file_name": r'H:\EXP2\FLAML\{}log.log'.format(m[0]),  # 日志输出路径
        "estimator_list": m,  # 指定使用的模型列表（此处只使用一种）
        "time_budget": 60 * 60 * 6,  # 总体训练时长限制为6小时
        "eval_method": 'holdout',  # 使用留出法进行验证
        "split_ratio": 0.2,  # 验证集占比20%
        "model_history": True,  # 是否记录模型历史
        "verbose": 3,  # 输出详细程度等级
        "seed": 7654321,  # 随机种子用于复现实验结果
        "early_stop": 5,  # 提前停止条件
    }

    # 初始化AutoML实例
    aml = AutoML()

    # 开始训练过程
    aml.fit(X_train=df.iloc[:, :-1], y_train=df.iloc[:, -1], **automl_settings)

    # 保存训练完成的模型至本地磁盘
    with open(r"F:\AutoML\AutoML\FLAML\ML\{}.pkl".format(m[0]), "wb") as f:
        pickle.dump(aml, f, pickle.HIGHEST_PROTOCOL)


# 方案一：分别单独训练多个不同的基础模型
models = [['lgbm'], ['xgboost'], ['rf'], ['extra_tree'], ['histgb'], ['catboost'], ['kneighbor']]
for i, _ in enumerate(models):
    print(models[i])
    train_method(models[i])

# 方案二：同时训练多个模型并让FLAML从中选择最优组合
models = ['lgbm', 'xgboost', 'rf', 'extra_tree', 'histgb']

# 设置多模型联合训练的AutoML参数
automl_settings = {
    "metric": 'rmse',
    "task": 'regression',
    "log_file_name": 'H:\EXP2\FLAML\multimodels_log.log',
    "estimator_list": models,
    "time_budget": 60 * 60 * 6,
    "eval_method": 'holdout',
    "split_ratio": 0.2,
    "model_history": True,
    "verbose": 3,
    "seed": 7654321,
    "early_stop": 5,
}

# 初始化AutoML实例
aml = AutoML()

# 执行多模型联合训练
aml.fit(X_train=df.iloc[:, :-1], y_train=df.iloc[:, -1], **automl_settings)

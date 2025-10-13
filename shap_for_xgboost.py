import numpy as np
import xgboost as xgb
import pickle
import shap
import pandas as pd
import flaml
from sklearn.metrics import r2_score as r2

# 测试数据
test_data=np.load(r"H:\EXP2\Data\test.npy").astype(np.float32)
# 列名
cols=["Curvature","Aspect","Slope","Elevation","Manhole density","TWI","Land use","Rainfall"]
for i in range(len(cols),68):
    cols.append(f"Rain_{i}")
cols.append("Y")
# 转为dataframe
df=pd.DataFrame(test_data,columns=cols)

# load model
with open(r"H:\EXP2\FLAML\xgboost.pkl", 'rb') as f:
    model = pickle.load(f)

# 获取 XGBoost 模型
xgb_model = model.model.estimator
# 修改参数以启用 GPU
xgb_model.set_params(
    tree_method="gpu_hist",
    device="cuda:0"
)

# 验证精度
pred=xgb_model.predict(df.iloc[:,:-1].to_numpy())
print(r2(df.iloc[:,-1].to_numpy(),pred))
# 模型解释

explainer=shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(df.iloc[:,:-1].to_numpy())
# 保存
np.save(r"H:\EXP2\FLAML\shap.npy",shap_values)

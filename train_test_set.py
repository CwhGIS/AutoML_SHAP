from osgeo import gdal,osr
import glob
import mikeio
import gc
from tqdm import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

seed=7654321
# 读取静态空间栅格数据
ds_curvature=gdal.Open(r"H:\MIKE\DEM_Features\curvature1.tif")
ds_slope=gdal.Open(r"H:\MIKE\DEM\slope.tif")
ds_aspect=gdal.Open(r"H:\MIKE\DEM\aspect.tif")
ds_density=gdal.Open(r"H:\MIKE\DEM\kernel_density.tif")
ds_twi=gdal.Open(r"H:\MIKE\DEM_Features\SAGA Topographic Wetness Index.tif")
ds_lucc=gdal.Open(r"H:\MIKE\DEM\LUCC2.tif")
ds_dem=gdal.Open(r"H:\MIKE\DEM_Features\output_dem.tif")
# arr
arr_curvature=ds_curvature.ReadAsArray()
arr_slope=ds_slope.ReadAsArray()
arr_aspect=ds_aspect.ReadAsArray()
arr_density=ds_density.ReadAsArray()
arr_twi=ds_twi.ReadAsArray()
arr_lucc=ds_lucc.ReadAsArray()
arr_dem=ds_dem.ReadAsArray()

# 查看图像是否正确
fig,ax=plt.subplots(2,4)
ax[0,0].imshow(arr_curvature)
ax[0,1].imshow(arr_slope)
ax[0,2].imshow(arr_aspect)
ax[0,3].imshow(arr_density)
ax[1,0].imshow(arr_twi)
ax[1,1].imshow(arr_lucc)
ax[1,2].imshow(arr_dem)

plt.show()
# 统计最大最小值
for v in [arr_curvature,arr_aspect,arr_slope,arr_dem,arr_density,arr_twi,arr_lucc]:
    print(np.nanmax(v),np.max(v),np.nanmin(v),np.min(v))

# 合并数据
dem_feature=np.ones((1,arr_slope.shape[0],arr_slope.shape[1]))
for v in [arr_curvature,arr_aspect,arr_slope,arr_dem,arr_density,arr_twi,arr_lucc]:
    v=np.expand_dims(v,axis=0)
    dem_feature=np.concatenate((dem_feature,v),axis=0)
dem_feature=dem_feature[1:,:,:]
print(dem_feature.shape)

# 屏蔽无效值
dem_feature=np.nan_to_num(dem_feature)
nums=np.where(dem_feature==-99999)
dem_feature[nums[0],nums[1],nums[2]]=0
nums=np.where(dem_feature>3e+38)
dem_feature[nums[0],nums[1],nums[2]]=0

"""
function: train_test_files
files=glob.glob(r"E:\60Min\*60Min*MAX*.dfs2")
from sklearn.model_selection import train_test_split as TTS

train,test=TTS(files,train_size=0.75,random_state=2025)
"""

test_files=[r'E:\60Min\2A60Min167MAXA01.dfs2',
 r'E:\60Min\5A60Min700MAXA01.dfs2',
 r'E:\60Min\10A60Min382MAXA01.dfs2',
r'E:\60Min\20A60Min167MAXA01.dfs2',
 r'E:\60Min\50A60Min500MAXA01.dfs2',
 r'E:\60Min\100A60Min500MAXA01.dfs2',]

# 训练集合
files=[]
for f in glob.glob(r"E:\60Min\*60Min*MAX*.dfs2"):
    if f not in test_files:
        files.append(f)
# 列名
cols=["Curvature","Aspect","Slope","Elevation","Manhole density","TWI","Land use","Rainfall"]
for i in range(len(cols),68):
    cols.append(f"Rain_{i}")
cols.append("Y")

# 训练df
df = pd.DataFrame()
for f in tqdm(files):
    ds_temp = mikeio.read(f)['max H'].to_numpy().squeeze()
    ds_temp = np.flip(ds_temp, axis=0)  # Y
    ds_temp = np.expand_dims(ds_temp, axis=0).reshape(1, -1)

    ds_all = np.concatenate((dem_feature.reshape(7, -1), ds_temp), axis=0)

    nums = np.where((np.isnan(ds_temp) == False) & (ds_temp >= 0.003))  # 忽略掉NAN值以及极小的淹没深度

    ds_all = ds_all[:, nums[1]].transpose(1, 0)
    df1 = pd.DataFrame(ds_all)

    rain_file = r"H:\MIKE\Rainfall\Designs\{}.dfs0".format(f.split("\\")[-1].split("MAX")[0])
    rain = mikeio.read(rain_file).to_numpy()
    rain = np.repeat(np.expand_dims(rain, axis=0), ds_all.shape[0], axis=0).squeeze()

    df1 = pd.concat([df1.iloc[:, :-1], pd.DataFrame(rain), df1.iloc[:, -1]], axis=1)
    df1.columns = cols
    df = pd.concat([df, df1], axis=0)

# 测试df
df_test = pd.DataFrame()
for f in tqdm(test_files):
    ds_temp = mikeio.read(f)['max H'].to_numpy().squeeze()
    ds_temp = np.flip(ds_temp, axis=0)  # Y
    ds_temp = np.expand_dims(ds_temp, axis=0).reshape(1, -1)

    ds_all = np.concatenate((dem_feature.reshape(7, -1), ds_temp), axis=0)

    nums = np.where((np.isnan(ds_temp) == False) & (ds_temp >= 0.003))  # 忽略掉NAN值以及极小的淹没深度

    ds_all = ds_all[:, nums[1]].transpose(1, 0)
    df1 = pd.DataFrame(ds_all)

    rain_file = r"H:\MIKE\Rainfall\Designs\{}.dfs0".format(f.split("\\")[-1].split("MAX")[0])
    rain = mikeio.read(rain_file).to_numpy()
    rain = np.repeat(np.expand_dims(rain, axis=0), ds_all.shape[0], axis=0).squeeze()

    df1 = pd.concat([df1.iloc[:, :-1], pd.DataFrame(rain), df1.iloc[:, -1]], axis=1)
    df1.columns = cols
    df_test = pd.concat([df_test, df1], axis=0)

df = shuffle(df)
np.save(r"H:\EXP2\Data\train.npy",df.to_numpy())
np.save(r"H:\EXP2\Data\test.npy",df_test.to_numpy())



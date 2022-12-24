import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## For preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
## 我們所使用的Machine Learning Algorithm
from xgboost import XGBRegressor
## For Evaluation Model
from sklearn.metrics import r2_score

data = pd.read_csv('./bmi_data.csv')

#將Sex資料取出並刪除該columns
sex = data[['Sex']]
data.drop(columns='Sex', axis=1, inplace=True)

#將data中有空缺的值進行中位數插補
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
Data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Sex變成無序型的類別型資料
OneHot = OneHotEncoder(sparse=False)
SEX = pd.DataFrame(OneHot.fit_transform(sex), columns=OneHot.get_feature_names_out())

# 將BMI資料取出並刪除該columns
target = Data['BMI']
Data.drop(columns='BMI', axis=1, inplace=True)

# 把最後的data做標準化
scale = StandardScaler()
scaleData = pd.DataFrame(scale.fit_transform(Data), columns=Data.columns)

#把SEX,scaleData的所有資料分成train和test
newData = pd.concat([SEX, scaleData], axis=1)
train_data,test_data,train_target,test_target\
        =train_test_split(newData, target, test_size=0.3, random_state=42)

#使用XGboost，並判斷正確度
XGB = XGBRegressor(verbosity=0)
XGB.fit(train_data, train_target)

#將資料分成五組並做交叉驗證
xgb_cv = cross_val_score(XGB, newData, target, cv=5)
print('Cross validation scores: ', xgb_cv)
print('Cross validation mean: ', np.mean(xgb_cv))

#預測testdata的prediction並和test_target做比較
prediction = XGB.predict(test_data)
print('predict score: ',r2_score(test_target, prediction))


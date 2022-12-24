import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.linear_model import LinearRegression,SGDRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

#讀取SimpleWeather.csv內的資料
data_path = "./bmi_data.csv"
df = pd.read_csv(data_path)

mean_Height = df[['Sex','Height']].groupby('Sex').mean()
Height_mapping = dict(zip(mean_Height.index, mean_Height['Height']))
for i in range(len(df['Height'].isnull())):
    if df['Height'].isnull()[i]:
        df.loc[i,'Height'] = Height_mapping.get(df['Sex'][i])

mean_Weight = df[['Sex','Weight']].groupby('Sex').mean()
Weight_mapping = dict(zip(mean_Weight.index, mean_Weight['Weight']))
for i in range(len(df['Weight'].isnull())):
    if df['Weight'].isnull()[i]:
        df.loc[i,'Weight'] = Weight_mapping.get(df['Sex'][i])
mean_BMI = df[['Sex','BMI']].groupby('Sex').mean()
BMI_mapping = dict(zip(mean_BMI.index, mean_BMI['BMI']))
for i in range(len(df['BMI'].isnull())):
    if df['BMI'].isnull()[i]:
        df.loc[i,'BMI'] = BMI_mapping.get(df['Sex'][i])

#把性別變成有序標籤
sex_en = LabelEncoder()
df['Sex'] = sex_en.fit_transform(df['Sex'].values)

#將資料分成data和target
data = df.values[::,0:4]
target = df.values[::,4]

#將data和target隨機分割成test和train 
train_data,test_data,train_target,test_target\
    =train_test_split(data,target,test_size=0.25,random_state=13)

#標準化
std_traindata = StandardScaler().fit_transform(train_data)
std_testdata = StandardScaler().fit_transform(test_data)
std_traintarget = StandardScaler().fit_transform(train_target.reshape(-1,1))
std_testtarget = StandardScaler().fit_transform(test_target.reshape(-1,1))

#建立線性預測模組，並判斷正確度
LR = LinearRegression()
LR.fit(std_traindata, std_traintarget)
print("the value of default measurement of linear regression"
                    ,LR.score(std_traindata,std_traintarget))

#線性預測traindata和testdata並和traintarget以及testtarget做比較
train_pred = LR.predict(std_traindata)
test_pred = LR.predict(std_testdata)
print("MSE train:%6f,test:%6f" %(mean_squared_error(std_traintarget,train_pred)
                                ,mean_squared_error(std_testtarget,test_pred)))

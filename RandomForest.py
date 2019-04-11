# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 21:35:09 2018

@author: alway
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

train=pd.read_csv('D:/Kaggle/data/train.csv')
test=pd.read_csv('D:/Kaggle/data/test.csv')
#train.isnull().sum().sort_values(ascending=True)

dtype_df = train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
unique_df = train.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
unique_df.head(10)
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape
train.drop(columns=constant_df['col_name'],inplace=True)
test.drop(columns=constant_df['col_name'],inplace=True)
print("updated train dataset shape",train.shape)
print("update test dataset shape", test.shape)
train.drop("ID", axis = 1, inplace = True)
y_train=train['target']
train.drop("target", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)
plt.figure(figsize=(10,6))
sns.distplot(y_train,kde=False, bins=20).set_title('Histogram of target');
plt.xlabel('Target')
plt.ylabel('count');
plt.figure(figsize=(10,6))
sns.distplot(np.log1p(y_train), bins=20,kde=False).set_title('Log histogram of Target');
minmax=MinMaxScaler()
x_train=minmax.fit_transform(train)
x_test=minmax.transform(test)
y_train=np.log1p(y_train)
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.05, random_state=0)
X_train.shape
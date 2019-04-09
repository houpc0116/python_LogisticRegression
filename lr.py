#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:22:23 2019

@author: hou
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LogisticRegression as LR
#載入資料
data=pd.read_csv("training.csv")
test=pd.read_csv("testing.csv")
#print(test.head(20))
#reshape 變成一列
x=data['value'].values.reshape(-1,1)  #特徵 - Training X
testX=test['value'].values.reshape(-1,1)  #特徵 - Testing X
#print(testX)
#print(testX.tolist())
"""
for i in range(len(testX)):
    print(testX[i][0])
"""
y=data['status'].values.reshape(-1,1)  #特徵 - Training Y
print(type(x), x.shape)
print(type(y), y.shape)
#print(x)

model=LR()
"""
訓練模型
"""
my_sccore=model.fit(x,y)    #建立模型
#print(my_sccore)
testY=model.predict(testX)  #預測testing value return result
print(testY)
#print("長度:", len(testY))
"""
建立二維串列
"""
list1 = []
for i in range(len(testY)):
    listX = []           #建立空串列
    listX.append(testX[i][0])  #test Value值
    listX.append(testY[i])     #test 預測
    list1.append(listX)        #建立

print(list1)
"""
寫入CSV檔案, 另產生檔案
"""    
with open('output.csv', 'w', newline='') as csvfile:
  # 定義欄位
  fieldnames = ['value', 'status']
  # 將 dictionary 寫入 CSV 檔
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  # 寫入第一列的欄位名稱
  writer.writeheader()
  # 建立 CSV 檔寫入器
  #寫入資料
  writer = csv.writer(csvfile)
  writer.writerows(list1)
  
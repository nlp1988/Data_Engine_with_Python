# -*- coding: utf-8 -*-

#Created on Mon Jul 06 15:27:42 2020

#@author: nlp1988

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from efficient_apriori import apriori


# 数据加载

data = pd.read_csv('./datasets_8127_11403_Market_Basket_Optimisation.csv', header = None)
#print('-'*20, '数据的行数和列数', '-'*20)
#print(data.shape)
#print(data.shape[1])

# 将数据存放到transactions中
transactions = []
  #遍历所有行
for i in range(0, data.shape[0]):
    temp = []
    #遍历所有列
    for j in range(0, data.shape[1]):
    # != 'nan'代表有数据
        if str(data.values[i, j]) != 'nan':
           temp.append(str(data.values[i, j]))
    transactions.append(temp)
    
#name=['shrimp', 'almonds', 'avocado', 'vegetables mix', 'green grapes', 'whole weat flour', 'yams', 'cottage cheese', 'energy drink', 'tomato juice', 'low fat\
#test=pd.DataFrame(columns=name,data=transactions)#数据有三列，列名分别为one,two,three
#print(test)
#test.to_csv('./test.csv',encoding='gbk')

print(transactions)
# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.4)
print('-'*20,'频繁项集','-'*20)
print("频繁项集：", itemsets)
print('-'*20, '关联规则', '-'*20)
print("关联规则：", rules)

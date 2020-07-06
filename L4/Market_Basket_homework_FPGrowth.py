# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:27:42 2020

@author: nlp1988
"""
import pandas as pd
import numpy as np
from pandas import Series, DataFrame  
import matplotlib.pyplot as plt
from sklearn import datasets # 机器学习库
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
import gc
import time
from sklearn.metrics import roc_auc_score

#===============================定义FP树类==========================
'''
#FP树需要的信息：子节点位置、父节点位置、计数、节点的一项集名称
'''
class FPtreeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.nameValue = nameValue
        self.numOccur = numOccur
        self.parentNode = parentNode
        self.sonNode = []  #因为创建节点时，很可能还没有子节点，所以要先把子节点设置成空


#==========读取数据
def read_data(table_str,col_name):
    data = pd.read_csv(table_str,encoding='gbk')
    data.fillna(-1)
    data = data[col_name]
    data = data.iloc[0:100,:]
    for i in col_name:
        data[i] = data[i].apply(lambda x:str(i)+str(x))
    return data

#==========找出频繁1项集，建立头指针表
'''
过程：1. 计算每个集合出现的频率
      2. 计算每个1项集出现的频率
'''
#用于读取DataFrame型的数据
def initFreq(data):
    initFreq = {}
    for i in range(data.shape[0]):
        if frozenset(data.iloc[i,:]) in list(initFreq.keys()):  # 这样每次都要将字典key组装成列表，字典可以直接 if key_obj in dict
            initFreq[frozenset(data.iloc[i,:])] = initFreq[frozenset(data.iloc[i,:])]+1
        else:
            initFreq[frozenset(data.iloc[i,:])] = 1  #为什么这里用list不行，用frozenset就可以？因为key值必须是不可修改类型。
    return initFreq

def initFreq_list(data):
    initFreq = {}
    for i in data:
        if frozenset(i) in initFreq.keys():  # 这样每次都要将字典key组装成列表，字典可以直接 if key_obj in dict
            initFreq[frozenset(i)] = initFreq[frozenset(i)]+1
        else:
            initFreq[frozenset(i)] = 1  #为什么这里用list不行，用frozenset就可以？因为key值必须是不可修改类型。
    return initFreq

def buildHeader(initFreq,minSup):
    headerTable = {}
    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',initFreq)
    for i in initFreq:
        #print('i',i)
        for j in i:
            if j in list(headerTable.keys()): # 直接in dict就可以。
                headerTable[j] = [headerTable[j][0]+initFreq[i],None]
            else:
                headerTable[j] = [initFreq[i],None]
    #取满足最小支持度的频繁一项集
    #这个部分比较棘手，如果直接用列表推导式，总是只能生成多个列表或字典组合成的列表
    keys = list(headerTable.keys())  #如果不加这句话，就会出现迭代过程中字典大小改变的错误
    for i in keys:
        if headerTable[i][0] < minSup:
            del(headerTable[i])
    headerTable = dict(sorted(headerTable.items(),key=(lambda x:x[1]),reverse=True))
    return headerTable

#=============================更新头指针，这样找条件模式基的时候才方便================
def updateHeader(headerTable,FPtreeNode):
    headerTable[FPtreeNode.nameValue].append(FPtreeNode)

def UpdateTree(treeNodeListTotal,headerTable,parentNode,currentNodeName,initNodeNum):
    sonNodeList = {}
    #先把key对应的节点的子节点取出来
    for i in parentNode.sonNode:
        sonNodeList[i.nameValue] = i
    #判断是否已经存在了相邻的子节点，如果有，则直接把numOccur+1
    if currentNodeName in sonNodeList.keys():
        sonNodeList[currentNodeName].numOccur = sonNodeList[currentNodeName].numOccur+1
        tempNode = sonNodeList[currentNodeName]
    #如果没有，则新增一个节点
    else:
        tempNode = FPtreeNode(currentNodeName,initNodeNum,parentNode)  #这里必须注意initNodeNum不能直接设为1，不然当出现多个重复节点需要创建时，计数就会不对
        parentNode.sonNode.append(tempNode)
        treeNodeListTotal.append(tempNode)
        updateHeader(headerTable,tempNode)
    return tempNode


def buildFptree(headerTable,data):
    #rootSon = []
    '''
    #建FP树的过程：
    1. 逐行读取输入的数据，在头指针表中把这些项集对应的支持度找出来
    2. 根据支持度进行排序，依次建立节点
    '''
    treeNodeListTotal = []  #记录整个FP树的结构
    rootNode = FPtreeNode('Null', 1, [])   #定义FP树的根节点
    #print(headerTable)
    for item in data:
        treeNodeList = {}  #用来记录本item中每个节点的指针
        treeTable = {}  #因为对于每个item，都应该是一个新的table，所以不在for循环外面定义
        for i in headerTable.keys():
            if i in item:
                treeTable[i] = headerTable[i][0]
        #print(treeTable)
        #对于treeTable中的每个节点，逐步建立树，先判断是否为首个节点，再判断是否需要新建节点，还是+1。
        #如果是首个节点，则判断根节点的子节点中是否有它，如果不是，则判断上一个节点的子节点中是否有它
        for i in range(len(treeTable.keys())):
            keyList = list(treeTable.keys())
            if i==0:
                treeNodeList[keyList[i]] = UpdateTree(treeNodeListTotal,headerTable,rootNode,keyList[i],data[item])
            else:
                treeNodeList[keyList[i]] = UpdateTree(treeNodeListTotal,headerTable,treeNodeList[keyList[i-1]],keyList[i],data[item])
        #检查建树过程：

        #print(treeNodeList)
#        for i in treeNodeList.keys():
#                print('打印建树过程',data,item,treeNodeList[i].nameValue,treeNodeList[i].numOccur,treeNodeList[i].parentNode)
#    for i in treeNodeListTotal:
#       print('FP树节点打印',i.nameValue,i.numOccur,i.parentNode)
    return treeNodeListTotal


#寻找前缀路径
def findParent(node,path):
    if node.parentNode!=None and node.parentNode.nameValue!='Null':
        path.append(node.parentNode.nameValue)
        findParent(node.parentNode,path)
    elif node.parentNode.nameValue=='Null' and len(path)==0:
        path.append([])
    #print(node.nameValue,path)
    return path

#==========找出条件模式基，创建条件FP树=================================
def findModeBase(headerTable):
    prePath = {}   #新建一个字典记录所有一项集的前缀路径
    for i in headerTable.keys():
        prePathNode = {}  #记录单个一项集的前缀路径
        for item in headerTable[i][2:]:
            path = []
            path = findParent(item,path)
            if [] not in path and len(path)>0:
                prePathNode[frozenset(path)] = item.numOccur
            else:
                prePathNode[frozenset([])] = item.numOccur
        #print(prePathNode,i,'------------------------------开始建条件FP树')
        tempHeader = buildHeader(prePathNode,1)
        #print('条件FP头',tempHeader)
        treeNodeListTotal = buildFptree(tempHeader,prePathNode)
        prePath[i] = prePathNode
        #print('前缀路径',prePath)
        return tempHeader,prePath,treeNodeListTotal

def findBaseFreq(baseFreq,tempHeader,minSup,freqList):
   # print('tempHeader',tempHeader,baseFreq)
    tempHeader_1,prePath,treeNodeListTotal = findModeBase(tempHeader)
   # print('tempHeader_1',tempHeader_1)
    if len(tempHeader_1.keys()) > 0:
      #  print('aaaaaaa')
        for i in tempHeader_1:
            tempFreq = baseFreq.copy()
            if tempHeader_1[i][0] >= minSup:
                if i not in tempFreq:
                    tempFreq.append(i)
                tempFreq_1 = tempFreq.copy()
                if not tempFreq in freqList:
                    freqList.append(tempFreq_1)
                   # print(tempFreq)
                   # print('freqList', freqList)

                    tempHeader_2 = {i: tempHeader_1[i]}
                   # print('tempHeader_2',tempHeader_2)
                    findBaseFreq(tempFreq.copy(),tempHeader_2,minSup,freqList)
   # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',baseFreq)

def findFreq(headerTable,minSup,freqList):
    tempHeader = {}
    for i in headerTable.keys():
        if i not in freqList and headerTable[i][0]>=minSup:
            freqList.append(i)
    #print('========================================',freqList)
   # print(headerTable,'\n\n\n\n')
    for i in headerTable:
        tempHeader[i] = headerTable[i]  # 列表引用复制
        baseFreq = [i]
      #  print('\n\n\n\n\n\n\n\n\n开始递归查找这个一项集的所有条件模式基',i,tempHeader[i])
        findBaseFreq(baseFreq,tempHeader,minSup,freqList)
        tempHeader = {}

      #  print('频繁项列表',freqList)
    return freqList



if __name__=='__main__':
    minSup = 110

    data = pd.read_csv('./datasets_8127_11403_Market_Basket_Optimisation.csv', header = None)
    #print('-'*20, '数据的行数和列数', '-'*20)
   # print(data.shape)
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

    
initFreq = initFreq_list(transactions)
  

headerTable = buildHeader(initFreq,minSup)
   # print('---------------------',headerTable)
buildFptree(headerTable,initFreq)
   # print(headerTable)
    #findModeBase(headerTable)
freqList = []
    #print('Fp树头指针',headerTable)
freqList = findFreq(headerTable,minSup,freqList)
    #print(i)
    
print(freqList)
#print(i)
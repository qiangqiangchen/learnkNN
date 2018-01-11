# -*- coding:utf-8 -*-

import numpy as np
import operator

"""
函数说明：创建数据集
Parameters:
    无
returns:
    grop-数据集
    labels-分类标签
"""
def creatDatSet():
    #四组二维特征
    group=np.array([[1,101],[5,89],[108,5],[115,8]])
    #四组特征的标签
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels

"""
函数说明：kNN算法，分类器
Parameters:
    inX-用于分类的数据（测试集）
    dataSet-用于训练的数据（训练集）
    labels-分类标签
    k-kNN算法参数，选择距离最小的k个点
returns:
    sortedClassCount[0][0]-分类结果
"""
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    
    





if __name__=="__main__":
    #创建数据集
    group,labels=creatDatSet()
    print(group)
    print(labels)
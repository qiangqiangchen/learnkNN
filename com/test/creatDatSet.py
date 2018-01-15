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
    inX:用于分类的数据（测试集）
    dataSet:用于训练的数据（训练集）
    labels:分类标签
    k:kNN算法参数，选择距离最小的k个点
returns:
    sortedClassCount[0][0]-分类结果
"""
def classify0(inX,dataSet,labels,k):
    #返回shape[0]的行数
    dataSetSize=dataSet.shape[0]
    #在列向量方向上重复inX共1次（横向），行向量方向上重复inX共dataSetSize次（纵向）
    diffMat=np.tile(inX, (dataSetSize,1))-dataSet
    #二维特征相减后平方
    sqDiffMat=diffMat**2
    #sum（）所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances=sqDiffMat.sum(axis=1)
    #开方，计算出距离
    distances=sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices=distances.argsort()
    #定一个记录类别次数的字典
    classCount={}
    for i in range(k):
        #取出前K个元素的类别
        voteIlabel=labels[sortedDistIndices[i]]
        #字典的get()方法，返回指定键的值，如果值不在字典中返回默认值。
        #记录类别次数
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
        
        
    #reverse降序排序字典
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    
    #返回次数最多的类别，即所要分类的类别
    return sortedClassCount[0][0]
    
    





if __name__=="__main__":
    #创建数据集
    group,labels=creatDatSet()
    #print(group)
    #print(labels)
    sortresult=classify0([101,20], group, labels, 3)
    print(sortresult)
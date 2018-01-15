# -*- coding:utf-8 -*-

import numpy as np
import os
import operator



"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量

"""

def img2vector(filename):
    #创建1*1024零向量
    returnVect=np.zeros((1,1024))
    #打开文件
    fr=open(filename)
    #按行读取
    for i in range(32):
        #读一行数据
        lineStr=fr.readline()
        #每一行的前32个元素依次添加到returnVect
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    #返回转换后的1*1024向量
    return returnVect



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



"""
函数说明:手写数字分类测试
Parameters:
    无
Returns:
    无
"""
def handwritingClassTest():
    #测试集的标签lables
    hwLabels=[]
    #返回trainingDigits下的文件名
    trainingFileList=os.listdir("trainingDigits")
    #返回trainingDigits下的文件个数
    m=len(trainingFileList)
    #训练的mat矩阵
    trainingMat=np.zeros((m,1024))
    #从文件名解析出训练的类别
    for i in range(m):
        fileNameStr=trainingFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:]=img2vector('trainingDigits/%s'%(fileNameStr))
    #测试集的文件列表
    testFileList=os.listdir('testDigits')
    #错误检测计数
    errorCount=0.0
    #测试数据的数量
    mTest=len(testFileList)
    #从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        fileNameStr=testFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s'%(fileNameStr))
        classifiterResult=classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类返回结果为 %d\t真实结果为%d "%(classifiterResult,classNumber))
        if(classifiterResult!=classNumber):
            errorCount+=1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest))
        
        
if __name__=='__main__':
    handwritingClassTest()
    

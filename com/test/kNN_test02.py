# -*- coding:utf-8 -*-

import numpy as np


def file2matrix(filename):
    #打开文件
    fr=open(filename)
    #读取文件所有内容
    arrayOLines=fr.readlines()
    #得到文件行数
    numberOfLines=len(arrayOLines)
    #返回的NumPy矩阵，解析完成的数据：arrayOLines列，3列
    returnMat=np.zeros((numberOfLines,3))
    #返回的分类标签向量
    classLabelVector=[]
    #行的索引值
    index=0
    #
    for line in arrayOLines:
        #删除字符串中的空白符号
        line=line.strip()
        #根据“\t”进行字符串分割
        listFormLine=line.split('\t')
        #将数据前三列提取出来，存放到returnMat的NumPy矩阵中，也就是特征矩阵
        returnMat[index,:]=listFormLine[0:3]
        #根据文本标记进行分类，1--不喜欢，2--魅力一般，3--极具魅力
        if listFormLine[-1]=='didntLike':
            classLabelVector.append(1)
        elif listFormLine[-1]=='smallDoses':
            classLabelVector.append(2)
        elif listFormLine[-1]=='largeDoses':
            classLabelVector.append(3)
            
        index+=1
        
    return returnMat,classLabelVector


if __name__=='__main__':
    
    datingDataMat,datingLabels=file2matrix("datingTestSet.txt")
    print(datingDataMat)
    print(datingLabels)
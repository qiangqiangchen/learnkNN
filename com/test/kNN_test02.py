# -*- coding:utf-8 -*-

import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import operator



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
    函数说明：打开并解析文件，对数据进行分类：1代表不喜欢，2代表魅力一般，3极具魅力
    parameters:
        filename:文件名
    Returns:
        returnMat:特征矩阵
        classLableVector-分类Label向量
"""
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

"""
    函数说明：数据可视化
    parameters:
        datingDataMat:特征矩阵
        classLableVector-分类Label向量
    Returns:
                        无
        
"""

def showdatas(datingDataMat,datingLabels):
    #设置字体
    font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)
    #设置画布,不共享xy轴，画布大小为（13，8）
    #当nrows=2, ncols=2，代表画布被划分为四个区域，axs[0][0]代表第一个区域
    fig,axs=plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, 
                         figsize=(13,13))
    numberOfLabels=len(datingLabels)
    LabelsColors=[]
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    #画出散点图，以datingDataMat矩阵的第一和第二列数据画散点数据，散点大小为15，透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelsColors,s=15,alpha=.5)
    
    #设置标题,x轴label,y轴label
    axs0_title_text=axs[0][0].set_title('每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text=axs[0][0].set_xlabel('每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text=axs[0][0].set_ylabel('玩视频游戏所消耗时间占',FontProperties=font)
    plt.setp(axs0_title_text,size=9,weight='bold',color='red')
    plt.setp(axs0_xlabel_text,size=7,weight='bold',color='black')
    plt.setp(axs0_ylabel_text,size=7,weight='bold',color='black')

    #画出散点图，以datingDataMat矩阵的第一和第三列数据画散点数据，散点大小为15，透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=.5)
    
    axs1_title_text=axs[0][1].set_title('每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text=axs[0][1].set_xlabel('每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text=axs[0][1].set_ylabel('每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text,size=9,weight='bold',color='red')
    plt.setp(axs1_xlabel_text,size=7,weight='bold',color='black')
    plt.setp(axs1_ylabel_text,size=7,weight='bold',color='black')
    
    #画出散点图，以datingDataMat矩阵的第二和第三列数据画散点数据，散点大小为15，透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=.5)
    
    axs2_title_text=axs[1][0].set_title('玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text=axs[1][0].set_xlabel('玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text=axs[1][0].set_ylabel('每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text,size=9,weight='bold',color='red')
    plt.setp(axs2_xlabel_text,size=7,weight='bold',color='black')
    plt.setp(axs2_ylabel_text,size=7,weight='bold',color='black')
    
    #设置图例
    didntLike=mlines.Line2D([],[],color='black',marker='.',markersize=6,label='didntLike')
    smallDoses=mlines.Line2D([],[],color='orange',marker='.',markersize=6,label='smallDoses')
    largeDoses=mlines.Line2D([],[],color='red',marker='.',markersize=6,label='largeDoses')
    
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    
    #显示图片
    plt.show()



"""
   函数说明：对数据进行归一化
    parameters:
        dataSet:特征矩阵
    Returns:
        normDataSet:归一化后的特征矩阵
        ranges:数据范围
        minVals:数据最小值
"""

def autoNorm(dataSet):
    #获得数据的最小值和最大值,min()无参，所有中的最小值，min(0)每列的最小值，min(1)每行的最小值
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    #最大值和最小值的范围
    ranges=maxVals-minVals
    #shape(dataSet)返回dataSet的矩阵行列数
    normDataSet=np.zeros(np.shape(dataSet))
    
    #返回dataSet行数
    m=dataSet.shape[0]
    
    #原始值减去最小值
    normDataSet=dataSet-np.tile(minVals, (m,1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet=normDataSet/np.tile(ranges, (m,1))
    
     #返回归一化数据结果,数据范围,最小值
    return normDataSet,ranges,minVals

def datingClassTest():
    datingDataMat,datingLabels=file2matrix("datingTestSet.txt")
    #获取所有数据的百分之十
    hoRatio=0.10
    #数据归一化，返回归一化后的矩阵
    normDataSet,ranges,minVals=autoNorm(datingDataMat)
    #获取归一化后的矩阵行数
    m=normDataSet.shape[0]
    #测试数据
    numTestVecs=int(m*hoRatio)
    #出错数
    errorCount=0.0
    
    for i in range(numTestVecs):
        #k-NN进行分类处理
        classifierResult=classify0(normDataSet[i,:], normDataSet[numTestVecs:m,:], datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真是类别:%d"%(classifierResult,datingLabels[i]))
        #错误数据统计
        if classifierResult!=datingLabels[i]:
            errorCount+=1.0
    print("错误率:%f%%"%(errorCount/float(numTestVecs)*100))






if __name__=='__main__':
    
#     datingDataMat,datingLabels=file2matrix("datingTestSet.txt")
#     showdatas(datingDataMat, datingLabels)
#     normDataSet,ranges,minVals=autoNorm(datingDataMat)
#     print(normDataSet)
#     print(ranges)
#     print(minVals)
    datingClassTest()
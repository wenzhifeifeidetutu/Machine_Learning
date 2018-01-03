#k-近邻算法
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    #获得第二维的长度group.shape[0] 着为4 shape【1】为2

    dataSetSize = dataSet.shape[0]

    #numpy.tile([0,0],(2,1))#在列方向上重复[0,0]1次，行2次

    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    #获取平方
    sqDiffMat = diffMat**2

    #现在对于数据的处理更多的还是numpy。没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加
    #获取每行的和

    sqDistances = sqDiffMat.sum(axis=1)

    #距离公式开方
    distances = sqDistances**0.5

    #将array进行升序排列
    sortedDistances = distances.argsort()

    #将字典分为元组
    classCount = {}

    for i in range(k):
        voteIlable = labels[sortedDistances[i]]

        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    #iteritems()
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


#将文本转化为Numpy函数
def file_to_matrix(filename):
    fr = open(filename)
    #转换为列表
    arrayLines = fr.readlines()
    #获取行数
    numberOfLines = len(arrayLines)

    #固定每行3列
    returnMat = zeros((numberOfLines, 3))

    classLabelVetor = []

    index = 0

    for line in arrayLines:
        #去除回车
        line = line.strip()
        #以tab分割每一行
        listFromLine = line.split('\t')
        #[1,:]表示矩阵的第一个向量的第二维度全部
        returnMat[index,:] = listFromLine[0:3]
        #储存每行结尾
        classLabelVetor.append(int(listFromLine[-1]))

        index += 1


    return  returnMat, classLabelVetor
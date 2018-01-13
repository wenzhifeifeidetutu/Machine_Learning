import k_NNshuanfa
import matplotlib
import matplotlib.pyplot as plt


group, labels = k_NNshuanfa.createDataSet()

print(k_NNshuanfa.classify0([0, 0], group, labels, 3))

datingDateMat,datingLabels = k_NNshuanfa.file_to_matrix('datingTestSet2.txt')

print(datingDateMat)

print(datingLabels[0:20])

##图像显示

# while 1:
#     try:
#         a = input('请输入点x坐标')
#         b = input('请输入y坐标')
#         a = int(a)
#         b = int(b)
#         print('最接近与'+ k_NNshuanfa.classify0([a, b], group, labels, 3) + ' class')
#
#     except:
#         break


# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.scatter(datingDateMat[:, 1], datingDateMat[:, 2], 15.0 * k_NNshuanfa.array(datingLabels), 15.0 * k_NNshuanfa.array(datingLabels))
#
# plt.show()

normMat, ranges, minVals = k_NNshuanfa.autoNorm(datingDateMat)
print()
print(normMat)
print()
print(ranges)

print()

print(minVals)

k_NNshuanfa.datingClassTest()


k_NNshuanfa.classifyPerson()






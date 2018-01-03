import k_NNshuanfa

group, labels = k_NNshuanfa.createDataSet()

print(k_NNshuanfa.classify0([0, 0], group, labels, 3))

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



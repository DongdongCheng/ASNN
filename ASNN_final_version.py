"""
Accelerated Searching for Natural Neighbours

according to the requirements
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import timeit
import copy
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings("ignore")
from NaN_KD import intersection
from sklearn.manifold import SpectralEmbedding

##############################################################################
#tools
##############################################################################
#Core Algorithm
def NaN_Searching(X,dataSetName):
    """
    Parameters
    ----------

    Returns
    -------
    r :
    nb :
    NaN : standard natural neighbors
    NNr : r-nn neighbors (kNN, k=r)
    NNmu : saturated neighbors (kNN, k=max(nb))
    """
    N = len(X)
#----------------------------------------------确定搜索范围---------------------------------------------------------------
    #第一步时间
    #找每个点的1近邻，记录距离最大值为max_dist，以此为边长画正方形
    start = timeit.default_timer()
    tree = KDTree(X, leaf_size=1)  # create a KDTree for X
    dist1, idx1 = tree.query(X, k=2)
    # max_dist = np.mean(dist)
    # print(np.mean(dist))
    #[[[[[[[[[............................调参.......................
    max_dist = np.max(dist1)

#--------------------------------------------搜索给定半径的点的个数---------------------------------------------------------
    #第二步时间
    points = {}
    points_dic = {}
    for i in range(len(X)):
        indices = tree.query_radius([X[i, :]], r=max_dist)
        points[i] = indices[0]

    #转换数据格式，不需要计算时间
    for i in range(len(X)):
        points_dic[i] = points[i].tolist()

#-------------------------------------------------找max_dist内点的个数的均值------------------------------------------------
    points_sum = 0
    for i in range(N):
        points_sum = points_sum+len(points_dic[i])
    # [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[............................调参.......................
    points_mean =math.floor(points_sum / N)
    print('the mean of points is:',points_mean)

# -----------------------------------------------找出“偏远点”-------------------------------------------------------------
    #若点的个数小于points_mean就是偏远点
    pianyuandian = []
    pianyuandian_neighbor_temp = []
    for i in range(N):
        if len(points[i]) < 0.5*points_mean:
            pianyuandian.append(i)
            pianyuandian_neighbor_temp.append(points[i].tolist())
    pianyuandian_copy = copy.deepcopy(list(pianyuandian))
    #删除邻居列表中是自己的元素
    for i in range(len(pianyuandian)):
        if pianyuandian[i] in pianyuandian_neighbor_temp[i]:
            pianyuandian_neighbor_temp[i].remove(pianyuandian[i])
    #提取偏远点的邻居为列表
    pianyuandian_neighbor = [n for a in pianyuandian_neighbor_temp for n in a]
    pianyuandian = set(pianyuandian)
    pianyuandian_neighbor=list(set(pianyuandian_neighbor))
    pianyuandian_neighbor_data=X[pianyuandian_neighbor,:]
    r = 3
    while True:
        # print("the {}-th iteration".format(r))
        dist, idx = tree.query(pianyuandian_neighbor_data,k = r)
        k_neighbors_unique = idx[:,1:r].flatten()
        pianyuandian=pianyuandian-set(k_neighbors_unique)
        # intersection = pianyuandian & (set(k_neighbors_unique))
        # for j in intersection:
        #     pianyuandian.remove(j)
        if len(pianyuandian) == 0:
            break
        else:
            r = r+1
    end = timeit.default_timer()
    time_total = end-start
    print('time_total = ', time_total)
    supk = r
    print('supk = ', supk)

# # #-------------------------------------------------------画出偏远点--------------------------------------------------------
#
#     pianyuandian_X = []
#     pianyuandian_Y = []
#     ax = plt.axes()
#     plt.plot(X[:,0], X[:,1], '.', color='k')
#     # 按下标添加标记
#     # for i in range(N):
#     #     plt.text(X[i,0], X[i,1], str(i), fontsize='7')
#     plt.xticks([])  # 去掉x轴刻度值
#     plt.yticks([])  # 去掉y轴刻度值
#     for i in range(len(pianyuandian_copy)):
#         pianyuandian_X.append(X[pianyuandian_copy[i],0])
#         pianyuandian_Y.append(X[pianyuandian_copy[i],1])
#
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.plot(pianyuandian_X,pianyuandian_Y,'.',color = 'r')
#
#
#     for i in range(len(pianyuandian_copy)):
#         # circle = Circle(xy=(pianyuandian_X[i], pianyuandian_Y[i]), radius=max_dist, color='#FFFACD')
#         # ax.add_patch(circle)
#         # 圆的基本信息
#         # 1.圆半径
#         r = max_dist
#         # 2.圆心坐标
#         a, b = (pianyuandian_X[i], pianyuandian_Y[i])
#         # ==========================================
#         # 方法一：参数方程
#         theta = np.arange(0, 2 * np.pi, 0.01)
#         x = a + r * np.cos(theta)
#         y = b + r * np.sin(theta)
#         # fig = plt.figure()
#         # axes = fig.add_subplot(111)
#         ax.plot(x, y,color = 'y',linewidth='0.7')
#
#
#     plt.savefig('./Datasets/ASNN_pic/' + '1_'+dataSetName + '_ASNN.png', dpi=300)
#     plt.savefig('./Datasets/ASNN_pic/' + '1_'+dataSetName + '_ASNN.eps', dpi=300)
#     plt.show()


    # return supk, nb, NaN, kNN, RkNN

    return supk

if __name__ == '__main__':
################################################ 人工数据集###################################################
    dataSetName = 'complex8'
    data = np.loadtxt('./Datasets/' + dataSetName + '.txt', delimiter=',')
    X = data[:, 0:-1]
    # supk, nb, NaN, kNN, RkNN = NaN_Searching(X,dataSetName)
    supk = NaN_Searching(X,dataSetName)

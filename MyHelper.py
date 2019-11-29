import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

'''
计算各特征值的方法
#邻域分析，表面粗糙等
'''

#计算最小二乘平面及距离
def CaculateAverageSquareDistance(p):
    num = p.shape[0]
    B = np.zeros((p.shape[0],3))
    one = np.ones((p.shape[0],1))
    B[:,0] = p[:,0]
    B[:,1] = p[:,1]
    B[:,2] = one[:,0]
    l = p[:,2]
    BTB = np.matmul(B.T,B)
    BTB_1 = np.linalg.pinv(BTB)
    temp = np.matmul(BTB_1,B.T)
    result = np.matmul(temp,l)
    V  = np.matmul(B,result)-l
    sum = 0
    for i in range (0,V.shape[0]):
        sum = sum+V[i]**2
    return sum/V.shape[0]

#粗糙度
def CaculateRoughness(p):
    p1 = np.zeros((p.shape[0],3),float)
    p1[:,0] = p[:,0]
    p1[:,1] = p[:,1]
    p1[:,2] = p[:,2]
    #print(p1)

    neigh = NearestNeighbors(n_neighbors=8)
    neigh.fit(p1)
    #index = neigh.kneighbors([[1.04291550e+05,4.79228380e+05,-8.10000050e-01]],return_distance=False)

    sum = 0
    for i in range(p1.shape[0]):
        index = neigh.kneighbors([p1[i]],return_distance=False)
        avedis2 = CaculateAverageSquareDistance(p1[index].reshape(8,3))
        sum = sum+avedis2

    return sum/p1.shape[0]*100

'''
def PcaPartAna(p):
    p1 = np.zeros((p.shape[0],3),float)
    p1[:,0:3] = p[:,0:3]

    pca = PCA(n_components=3)
    newX = pca.fit_transform(p1)
    a =(pca.explained_variance_ratio_)
    return (a[2]/a[0])

def PCAPlaneAynalize(p):
    p1 = np.zeros((p.shape[0],3),float)
    p1[:,0] = p[:,0]
    p1[:,1] = p[:,1]
    p1[:,2] = p[:,2]
    #print(p1)

    neigh = NearestNeighbors(n_neighbors=20)
    neigh.fit(p1)
    #index = neigh.kneighbors([[1.04291550e+05,4.79228380e+05,-8.10000050e-01]],return_distance=False)

    sum = 0
    for i in range(p1.shape[0]):
        index = neigh.kneighbors([p1[i]],return_distance=False)
        avedis2 = PcaPartAna(p1[index].reshape(20,3))
        sum = sum+avedis2

    return sum/p1.shape[0]

    # p1 = np.zeros((p.shape[0],3),float)
    # p1[:,0:3] = p[:,0:3]

    # pca = PCA(n_components=3)
    # newX = pca.fit_transform(p1)
    # a =(pca.explained_variance_ratio_)
    # return (a[2]/a[0])
'''

#邻域分析
def NeiAna(p):
    p1 = p[:,0:3]

    sum = 0
    for i in range(0,p1.shape[0],3):
        neigh = NearestNeighbors(n_neighbors=20)
        neigh.fit(p1)
        index = neigh.kneighbors([p1[i]],return_distance=False)

        pp = p1[index].reshape(20,3)
        pp[:,0] = pp[:,0]-np.mean(p1[:,0])
        pp[:,1] = pp[:,1]-np.mean(p1[:,1])
        pp[:,2] = pp[:,2]-np.mean(p1[:,2])
        a = pp.T
        b  = np.matmul(pp.T,pp)
        fev = np.linalg.eigvals(b)
        fev = (np.sort(fev))
        sum = sum+(fev[0])/fev[2]*1000
    
    return (sum/p1.shape[0]/3)

#组成特征向量
def GetFeatureVector(p):
    fv = np.array([[0.0,0.0]],dtype=float)
    fv[0,0] = NeiAna(p)
    fv[0,1] = CaculateRoughness(p)
    return fv

import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
import MyHelper

p = np.loadtxt("./points/trees/t4.txt")
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
    avedis2 = MyHelper.CaculateAverageSquareDistance(p1[index].reshape(8,3))
    sum = sum+avedis2

print(sum/p1.shape[0])


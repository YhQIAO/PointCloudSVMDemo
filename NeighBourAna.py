import numpy as np
from sklearn.neighbors import NearestNeighbors

p= np.loadtxt("./points/trees/t1.txt")
p1 = p[:,0:3]


neigh = NearestNeighbors(n_neighbors=8)
neigh.fit(p1)
index = neigh.kneighbors([p1[0]],return_distance=False)

p1 = p1[index].reshape(8,3)
print(p1)
p1[:,0] = p1[:,0]-np.mean(p1[:,0])
p1[:,1] = p1[:,1]-np.mean(p1[:,1])
p1[:,2] = p1[:,2]-np.mean(p1[:,2])

print(p1)
# a = p1.T
# b  = np.matmul(p1.T,p1)
# print(b)

# print(np.linalg.eig(b))
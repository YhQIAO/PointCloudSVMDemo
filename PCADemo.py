import numpy as np
from sklearn.decomposition import PCA


print("-----trees-----")
for i in range(1,16):
    p= np.loadtxt("./points/buildings/b"+str(i)+".txt")

    p1 = np.zeros((p.shape[0],3),float)
    p1[:,0:3] = p[:,0:3]

    pca = PCA(n_components=3)
    newX = pca.fit_transform(p1)
    a =(pca.explained_variance_ratio_)
    # print(newX)
    print(a[2]/a[0])
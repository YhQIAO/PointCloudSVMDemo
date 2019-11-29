from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import MyHelper

fig = plt.figure()
ax1 = Axes3D(fig)

p= np.loadtxt("./points/buildings/b6.txt")

for i in range(p.shape[0]):
    ax1.scatter3D(p[i,0],p[i,1],p[i,2],c = 'b')

B = np.zeros((p.shape[0],3))
one = np.ones((p.shape[0],1))
B[:,0] = p[:,0]
B[:,1] = p[:,1]
B[:,2] = one[:,0]

l = p[:,2]

BTB = np.matmul(B.T,B)

BTB_1 = np.linalg.inv(BTB)
temp = np.matmul(BTB_1,B.T)
result = np.matmul(temp,l)

x = np.linspace(min(p[:,0]),max(p[:,0]))
y = np.linspace(min(p[:,1]),max(p[:,1]))
X,Y = np.meshgrid(x,y)
Z = result[0]*X+result[1]*Y+result[2]

print(Z)

ax1.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap='rainbow')

plt.show()
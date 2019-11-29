from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import MyHelper
import FileOperator

f = open("./FeatureVectors.txt",'a')
f.seek(0)
f.truncate()
f.close()

print("-----trees-----")
for i in range(1,16):
    p= np.loadtxt("./points/trees/t"+str(i)+".txt")
    print(MyHelper.GetFeatureVector(p))
    v = MyHelper.GetFeatureVector(p)
    plt.scatter(v[0,0],v[0,1],c = 'g')
    FileOperator.WriteData("./FeatureVectors.txt",v,1)
    

print("-----buildings-----")
for i in range(1,16):
    p= np.loadtxt("./points/buildings/b"+str(i)+".txt")
    print(MyHelper.GetFeatureVector(p))
    v = MyHelper.GetFeatureVector(p)
    plt.scatter(v[0,0],v[0,1],c = 'b')
    FileOperator.WriteData("./FeatureVectors.txt",v,-1)

plt.show()

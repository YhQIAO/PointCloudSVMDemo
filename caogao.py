from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import MyHelper
import FileOperator

print("-----trees-----")
for i in range(1,16):
    p= np.loadtxt("./points/trees/t"+str(i)+".txt")
    print(MyHelper.NeiAna(p))
    
    

print("-----buildings-----")
for i in range(1,16):
    p= np.loadtxt("./points/buildings/b"+str(i)+".txt")
    print(MyHelper.NeiAna(p))
   
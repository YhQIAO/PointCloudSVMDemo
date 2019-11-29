from sklearn import svm
import random
import numpy as np
from matplotlib import pyplot as plt
import MyHelper
import FileOperator
import os

data = np.loadtxt("./FeatureVectors.txt",dtype = float,delimiter=',')

'''
for i in range(data.shape[0]):
    if data[i,2] == 1:
        plt.scatter(data[i,0],data[i,1],c = 'b')
    else:
        plt.scatter(data[i,0],data[i,1],c = 'r')
'''

# svm training
x,y=np.split(data,indices_or_sections=(2,),axis=1)

clf = svm.SVC(kernel='rbf')
clf.fit(x,y)
print(clf)
#plt.show()

#print(clf.predict([[5.69866481e-05,1.02679807e+01]]))


pathStr = "./points/testdata/data3/pc_"
ResultPathStr = "./points/testdata/resultData/re_"

path = './points/testdata/resultData'
for i in os.listdir(path):
   path_file = os.path.join(path,i)  
   if os.path.isfile(path_file):
      os.remove(path_file)

for i in range(1,344):
    
    filepath = pathStr+str(i)+".txt"
    p= np.loadtxt(filepath)
    fv = MyHelper.GetFeatureVector(p)
    label = clf.predict(fv)
    resultPath = ResultPathStr +str(i)+".txt"
    FileOperator.WritePointCloudRed(resultPath,p,label)
    print("第"+str(i)+"个数据处理成功，标签为"+str(label))



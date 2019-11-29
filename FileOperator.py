import numpy as np
import MyHelper
import FileOperator

ResultPath = "./points/testdata/resultData/re_"

def WriteData(path,v,type):
    f = open(path,'a')
    f.writelines(str(v[0,0])+","+str(v[0,1])+","+str(type)+"\n")
    f.close()

def WritePointCloud(ResultPath,p,label):
    #ResultPath = "./points/testdata/resultData/re_"+str(i)+".txt"
    f = open(ResultPath,'a')
    
    if label == 1:
        #给树着色 颜色为绿色
        for j in range(0,p.shape[0]):
            f.writelines(str(p[j,0])+" "+str(p[j,1])+" "+str(p[j,2])+" 0 "+"255 "+"0"+"\n")
        f.close()

    else:
        #给建筑着色 颜色为蓝色
        for j in range(0,p.shape[0]):
            f.writelines(str(p[j,0])+" "+str(p[j,1])+" "+str(p[j,2])+" 0 "+"0 "+"255"+"\n")
        f.close()

def WritePointCloudRed(ResultPath,p,label):
    #ResultPath = "./points/testdata/resultData/re_"+str(i)+".txt"
    f = open(ResultPath,'a')
    
    if label == -1:
        #给树着色 颜色为绿色
        for j in range(0,p.shape[0]):
            f.writelines(str(p[j,0])+" "+str(p[j,1])+" "+str(p[j,2])+" 255 "+"0 "+"0"+"\n")
        f.close()

    else:
        #给建筑着色 颜色为蓝色
        for j in range(0,p.shape[0]):
            f.writelines(str(p[j,0])+" "+str(p[j,1])+" "+str(p[j,2])+" 0 "+"0 "+"0"+"\n")
        f.close()
# PointCloudSVMDemo
三维点云激光分类（建筑，树木）

# 1.样本数据选取

## 1.1.原始样本：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129221515512.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129221531760.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)

## 1.2 样本预处理
首先去除地面，（滤波，随机采样等均可），利用连通性分析分离点云，同时删除点云个数小于20个点的点云快。（剔除一些杂点，不太便于辨别形状的点）。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129223718269.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
![：](https://img-blog.csdnimg.cn/20191129223741925.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
## 1.3 训练样本数据选取
![建筑部分训练样本](https://img-blog.csdnimg.cn/20191129224037854.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
![树木部分训练样本](https://img-blog.csdnimg.cn/201911292241033.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)

# 2.特征值的选取与计算（基于表面邻域分析）
由于树木和建筑直观上看还是比较好区分的，因此取两个特征值组成特征向量分类效果还可以。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129224227878.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
## 2.1 粗糙度的计算
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129224343676.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129224351750.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
寻找平面计算方法很类似于平差里面里的间接平差
ps(一开始想偷懒直接对整体进行计算，对于树还是可以，但对于建筑就有坑）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129224703509.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
建筑如果是由坡面组成，直接对整体进行计算粗糙度是不科学的。
## 2.2 基于KNN表面邻域分析
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129224838864.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
在点云表面取一个点，找到距离最近的n个点，这样的话建筑表面取点，最近的n个点一般都会是平面，树木表面取最近的n个点的话，按上述方法计算出来的粗糙的比较大。

KNN算法很多库都有，C#里有Alglib，Python的一个强大的第三方库sklearn里面也有封装
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129225335860.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129225519432.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
算下来的话可以看到建筑和树木差异还是比较大的，可以作为一个较好的分类标准。
## 2.3 点云表面局部协方差矩阵分析
老师上课PPT里有的一个东西，一开始没怎么懂，后来在《遥感学报》上看到了一篇论文用到了这个来做分类，实践了下后懂了原理，就是通过计算协方差阵的特征值来描述点云的分布。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112922570799.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
【来源】：老师PPT
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129225941572.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129225947521.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
【来源】：老师PPT
![来源：遥感学报，马振宇，庞勇，李增元，卢昊，刘鲁霞，陈博伟.地基激光雷达森林近地面点云精细分类与倒木提取 [J].遥感学报，2019, 23(4)。](https://img-blog.csdnimg.cn/20191129225959407.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
【来源】遥感学报，马振宇，庞勇，李增元，卢昊，刘鲁霞，陈博伟.地基激光雷达森林近地面点云精细分类与倒木提取 [J].遥感学报，2019, 23(4)

如果面状性较好的话，可以知道会有两个特征值大小接近，一个特征值很小；如果点云很分散的话，那么3个特征值的大小应该是比较接近的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129230324257.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
这里取发散状指数进行计算，可以看出差异还是比较大的，效果比较好。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129230355639.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
把上面两个特征值作出特征空间（二维来说就是个平面），可以看出可分性还是比较强的，用一个线性SVM分类器分类效果还是可以。

# 3 支持向量机分类
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129230527973.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
原理省略n字。。。。反正很复杂，但是我们把它看成一个黑箱模型就够了，就是往里面扔特征向量，往外面吐分类结果就可以了。

我用的是python里Sklearn库里面SVM，sklearn也安利一下，基本常见的机器学习算法里面都有封装好。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129230656561.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129230750841.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
和函数里面可以选线性函数，高斯函数等，这里就直接线性了。

# 4 分类结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129230845786.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129230852187.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)

# 5. 评价
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191129231033765.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE4MDIzNg==,size_16,color_FFFFFF,t_70)

# 6.源代码
完整代码以及点云可以访问Github
[https://github.com/YhQIAO/PointCloudSVMDemo.git](https://github.com/YhQIAO/PointCloudSVMDemo.git)
## 6.1 说明
***程序结构说明***
依赖文件：
MyHelper.py：封装计算特征值和特征向量的函数。
FileOperator.py: 封装文件读取和写入功能。
程序运行：
1.首先运行SaveFV.py，读取./points/trees以及./points/buildings中的训练样本数据，计算特征值，设树木标签为1，建筑标签为-1。保存的文件位于./FeatureVectors.txt。
2.再运行svmdemo.py，训练支持向量机分类器，并且读取待分类数据（位于./points/testdata/data1文件夹下），若分类为树木，则在每一点坐标后添加0 255 0，使其在导入CloudCompare时颜色为绿色；若分类为建筑，则在每一点坐标后添加0 0 255，使其在导入CloudCompare时颜色为蓝色。

***测试步骤：***
1.运行SaveFV.py,提取样本数据的特征向量，保存于FeatureVectors.txt文件中。
2.运行svmdemo.py，训练分类器，并对待分类数据进行分类
3.在CloudCompare中查看分类结果（蓝色的表示分类为建筑，绿色表示分类为树木）

***文件路径格式说明
本程序运行在MacOS（类Unix，Linux）系统上，文件路径格式为./…/… ,若在windows环境下运行，需修改文件路径格式。如./points/trees需修改为D:\points\trees。***

***文件内容说明***
所有点云数据在./points文件夹内
./points/buildings内为建筑样本
./points/trees内为树木样本
./points/testdata/data1 为BlockA_FULLA.txt测试数据
./points/testdata/data2 为BlockB_FULL.txt测试数据
./points/testdata/resultdata 为分类完成后的点云数据

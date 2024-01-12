# 聚类模块README
## 1. 聚类模块介绍
### 1.1 聚类模块功能
聚类模块主要实现了对提取到的文本的的聚类，包括使用的聚类算法有KMeans聚类、DBSCAN聚类两种方式。

输入爬虫爬到的txt格式的文本，输出聚类结果，包括经过PCA降维之后的数据点云图，在终端输出聚类结果的标签。

配合前端app展示页面，可以调用聚类模块的接口，实现对文本的聚类。

### 1.2 聚类模块目录结构
```
cluster
├── README.md
├── __init__.py
├── dbscan.py
├── kmeans.py
├── evaluation.py
```
其中`dbscan.py`和`kmeans.py`分别实现了DBSCAN聚类和KMeans聚类算法，在这两个文件中分别封装了两个函数，即`Cluster_dbscan`、`draw_dbscan`和`Cluster_kmeans`、`draw_kmeans`，分别实现了聚类和绘制聚类结果的功能，其中`Cluster_dbscan`和`Cluster_kmeans`两者传入的参数都是影评文件的存储路径，而`draw_dbscan`和`draw_kmeans`两者传入的参数都是一个list,list中的每一个元素都是一条影评。


## 2. 聚类模块使用
### 2.1 聚类模块使用方法
聚类模块使用方法如下：

**DBSCAN**


直接使用：```python src\cluster\dbscan.py```可以调整参数，参数如下：```file_path, file_name, save_path ,eps, min_samples```

调用函数```Cluster_dbscan```，传入参数为影评文件的存储路径，输出聚类结果的标签，保存聚类结果点云图，参数结构为```file_path, file_name, save_path ,eps, min_samples```，```file_path```是影评文件的路径，```fle_name```是电影名，```save_path```是结果点云图的存储路径；调用函数```draw_dbscan```，传入参数为一个list，list中的每一个元素都是一条影评，返回聚类结果的标签和经过PCA降维之后的数据点云图，参数结构为：```comments_list, save_path```。

**KMeans**


直接使用：```python src\cluster\kmeans.py```可以调整参数，参数如下：```file_path, file_name, save_path ,n_clusters```

调用函数```Cluster_kmeans```，传入参数为影评文件的存储路径，输出聚类结果的标签，保存聚类结果点云图，参数结构为```eps=0.05, min_samples=3, file_path, file_name, save_path```，```file_path```是影评文件的路径，```fle_name```是电影名，```save_path```是结果点云图的存储路径；调用函数```draw_kmeans```，传入参数为一个list，list中的每一个元素都是一条影评，返回聚类结果的标签和经过PCA降维之后的数据点云图，参数结构为：```comments_list, save_path```。




---
title: Dive into Deep Learning 笔记及作业解答
author: Shuitata
top: false
hide: false
cover: true
toc: true
mathjax: true
tags:
-
date: 2023-01-30 18:41:07
img:
coverImg:
password:
summary: Dive into Deep Learning（动手学深度学习 https://zh.d2l.ai/index.html）的课程笔记与部分练习解答
categories: 
---
# 2.预备知识

## 2.1数据操作

### 练习1

> 运行本节中的代码。运行本节中的代码。将本节中的条件语句X == Y更改为X < Y或X >  Y，然后看看你可以得到什么样的张量。
```python
  X = torch.arange(12, dtype=torch.float32).reshape((3,4))
  Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```
```python
  X > Y
```
结果略

### 练习2

> 用其他形状（例如三维张量）替换广播机制中按元素操作的两个张量。结果是否与预期相同？

  与预期结果相同

  运行
  ```python
  a = np.arange(3).reshape((1, 3, 1))
  b = np.arange(4).reshape((2,1, 2)
  a, b
  ```
  输出为:
  ```python
  (tensor([[[0],
            [1],
            [2]]]),

  tensor([[[0, 1]],
  
          [[2, 3]]]))
  ```
  再运行
  ```python
  a+b
  ```
  结果为：
  ```python
  tensor([[[0, 1],
          [1, 2],
          [2, 3]],

          [[2, 3],
          [3, 4],
          [4, 5]]])
  ```

### 广播机制
要求进行运算的两个张量满足：**相应维度的大小要么相同，要么其中有一个为1.**

## 2.2 数据预处理

### 练习1

> 删除缺失值最多的列

对于数据：
```python
   NumRooms Alley   Price
0       NaN  Pave  127500
1       2.0   NaN  106000
2       4.0   NaN  178100
3       NaN   NaN  140000
```
方案1：
```python
data.isnull() #判断是否缺少值
# 输出为
NumRooms	Alley	Price
0	True	False	False
1	False	True	False
2	False	True	False
3	True	True	False

data.isnull().sum() #计算列中缺失总数
#或者： data.isnull().sum(axis=0)
#输出为
NumRooms    2
Alley       3
Price       0

data.isnull().sum().idxmax() #得到最大缺失值的索引
#输出为
‘Alley’

data.drop(data.isnull().sum().idxmax(),axis=1) #按列删除 ‘Alley’
#输出为
NumRooms	Price
0	NaN	127500
1	2.0	106000
2	4.0	178100
3	NaN	140000
```
注：若想删去缺失值最多的行可以data.sum()改为data.sum(axis=1)

方案2：
```python
data.count(axis='index') #按列得到非缺失值的个数，
#输出为
NumRooms    2
Alley       1
Price       4

data.drop(data.count(axis='index').idxmin(),axis=1)#不替换原data
#输出为
NumRooms	Price
0	NaN	127500
1	2.0	106000
2	4.0	178100
3	NaN	140000

data.drop(data.count(axis='index').idxmin(),axis=1,inplace=True) #替换原来的data
#输出为
NumRooms	Price
0	NaN	127500
1	2.0	106000
2	4.0	178100
3	NaN	140000
```
注：若想删去缺失值最多的行可以data.count(axis='index')改为data.count(axis='column')

### 练习2

> 将预处理后的数据集转换为张量格式
```python
import torch
a = torch.tensor(data.values)
a
#输出为
tensor([[       nan, 1.2750e+05],
      [2.0000e+00, 1.0600e+05],
      [4.0000e+00, 1.7810e+05],
      [       nan, 1.4000e+05]], dtype=torch.float64)
```

## 2.3 线性代数

### 练习3

> 给定任意方阵$\mathbf{A}$，$\mathbf{A}+\mathbf{A}^\text{T}$总是对称的吗？

$$ (\mathbf{A}+\mathbf{A}^\text{T})_{ij}=\mathbf{A}_{ij}+\mathbf{A}^\text{T}_{ij}=\mathbf{A}^\text{T}_{ji}+\mathbf{A}_{ji}=(\mathbf{A}+\mathbf{A}^\text{T})_{ji} $$

### 练习4 与 练习5

> 本节中定义了形状$(2,3,4)$的张量X。len(X)的输出结果是什么？
> 对于任意形状的张量X,len(X)是否总是对应于X特定轴的长度?这个轴是什么?

X的第0轴长度

### 练习6

> 运行A/A.sum(axis=1)，看看会发生什么。请分析一下原因？

会输出：
```python
RuntimeError: The size of tensor a (4) must match the size of tensor b (5) at non-singleton dimension 1
```
因为A.sum(axis=1)相较于A减少了第1轴，无法进行运算。

### 练习7

> 考虑一个具有形状$(2,3,4)$的张量，在轴0、1、2上的求和输出是什么形状?
```python
torch.Size([3,4])
torch.Size([2,4])
torch.Size([2,3])
```

### 练习8

> 为linalg.norm函数提供3个或更多轴的张量，并观察其输出。对于任意形状的张量这个函数计算得到什么?

所有元素平方和的平方根


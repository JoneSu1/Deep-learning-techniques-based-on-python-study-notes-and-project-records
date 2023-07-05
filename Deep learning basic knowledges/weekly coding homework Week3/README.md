### 实现一个具有单一隐藏层的2类分类神经网络###

- 使用具有非线性激活函数的单元，如tanh
- 计算交叉熵损失
- 实现前向和后向传播

- <a name='1'></a>
# 1 - Packages

First import all the packages that you will need during this assignment.

- [numpy](https://www.numpy.org/) is the fundamental package for scientific computing with Python.
- [sklearn](http://scikit-learn.org/stable/) provides simple and efficient tools for data mining and data analysis. 
- [matplotlib](http://matplotlib.org) is a library for plotting graphs in Python.
- testCases provides some test examples to assess the correctness of your functions
- planar_utils provide various useful functions used in this assignment

  首先导入所有你在这次作业中需要的包。

- numpy是用Python进行科学计算的基本包。
- sklearn为数据挖掘和数据分析提供了简单而有效的工具。
- matplotlib是一个用于在Python中绘制图形的库。
- testCases 提供了一些测试实例，以评估你的函数的正确性。
- planar_utils提供了本作业中使用的各种有用的函数


  **Coding**

        # Package imports
        import numpy as np
        import copy
        import matplotlib.pyplot as plt
        from testCases_v2 import *
        from public_tests import *
        import sklearn
        import sklearn.datasets
        import sklearn.linear_model
        from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

        %matplotlib inline

        %load_ext autoreload
        %autoreload 2

  <a name='2'></a>
# 2 - Load the Dataset 

        X, Y = load_planar_dataset()

使用matplotlib对数据集进行可视化。该数据看起来像一朵 "花"，有一些红色（标签y=0）和一些蓝色（y=1）的点。你的目标是建立一个模型来适应这个数据。
换句话说，我们希望分类器能将区域定义为红色或蓝色。

![11](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d987eef8-ce10-478c-b68d-e768e25673da)



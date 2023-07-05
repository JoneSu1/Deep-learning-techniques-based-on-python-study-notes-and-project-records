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

![12](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/5c910039-0b9e-4964-aa43-995ec9941fcf)

You have:

- a numpy-array (matrix) X that contains your features (x1, x2)
- a numpy-array (vector) Y that contains your labels (red:0, blue:1).
First, get a better sense of what your data is like.

### 判断X（特征）和Y(标签)的数量有多少

         # (≈ 3 lines of code)
         # shape_X = ...
         # shape_Y = ...
         # training set size
         # m = ...
         # YOUR CODE STARTS HERE
         shape_X = X.shape
         shape_Y = Y.shape
         m = X.shape[1]
         # YOUR CODE ENDS HERE

         print ('The shape of X is: ' + str(shape_X))
         print ('The shape of Y is: ' + str(shape_Y))
         print ('I have m = %d training examples!' % (m))

![13](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/5e8b8865-ba73-45a1-b3de-b83cc5ea7932)


### 3 - 简单逻辑回归
在构建完整的神经网络之前，我们首先看看逻辑回归如何解决这个问题。您可以使用 sklearn 的内置函数来做到这一点。运行下面的代码以在数据集上训练逻辑回归分类器。

![14](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d3c48807-7595-4bc1-b7de-d4243a2d2bbc)

![15](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/5a43732d-3c9a-4949-bd7a-bb7b63f28f75)

<a name='4'></a>
## 4 - Neural Network model

Logistic regression didn't work well on the flower dataset. Next, you're going to train a Neural Network with a single hidden layer and see how that handles the same problem.

**The model**:
![16](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/1f6225ac-b4c1-4984-b696-675bf59793a6)



**Reminder**: The general methodology to build a Neural Network is to:
    1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
    2. Initialize the model's parameters
    3. Loop:
        - Implement forward propagation
        - Compute loss
        - Implement backward propagation to get the gradients
        - Update parameters (gradient descent)


**提醒**： 构建神经网络的一般方法是：：
    1. 定义神经网络结构（输入单元的数量，隐藏单元的数量等）。
    2. 初始化模型的参数
    3. 循环：
        - 实施前向传播
        - 计算损失
        - 实施后向传播以获得梯度
        - 更新参数（梯度下降）

In practice, you'll often build helper functions to compute steps 1-3, then merge them into one function called `nn_model()`. Once you've built `nn_model()` and learned the right parameters, you can make predictions on new data.


在实践中，你通常会建立辅助函数来计算第1-3步，然后将它们合并到一个叫做`nn_model()`的函数中。一旦你建立了`nn_model()`并学习了正确的参数，你就可以对新的数据进行预测了。


<a name='4-1'></a>
### 4.1 - Defining the neural network structure ####

<a name='ex-2'></a>
### Exercise 2 - layer_sizes 

Define three variables:
    - n_x: the size of the input layer
    - n_h: the size of the hidden layer (**set this to 4, only for this Exercise 2**) 
    - n_y: the size of the output layer

**Hint**: Use shapes of X and Y to find n_x and n_y. Also, hard code the hidden layer size to be 4.

4.1 - 定义神经网络结构

练习2--层_尺寸
定义三个变量：

- n_x: 输入层的大小
- n_h：隐藏层的大小（**将其设为4，仅用于本练习2**）。
- n_y：输出层的大小。
- 
提示：使用X和Y的形状来寻找n_x和n_y。同时，将隐藏层的大小硬编码为4。

在神经网络中，输入层的神经元数量由特征数据集的维度决定。每个特征在神经网络中对应一个输入神经元。例如，如果特征数据集X的形状为(n_x, m)，其中n_x表示特征的数量，
m表示样本的数量，那么输入层的神经元数量就是n_x。

输出层的神经元数量通常由任务的要求决定。在分类问题中，输出层的神经元数量通常与类别的数量相同，每个输出神经元对应一个类别。例如，对于二分类问题，输出层有两个神经元，
分别表示两个类别。对于多分类问题，输出层的神经元数量等于类别的数量。

因此，输入层和输出层的神经元数量是根据问题的特性和要求来确定的，而隐藏层的神经元数量可以根据网络架构和实验需求进行设置。

**Code**

# GRADED FUNCTION: layer_sizes

        def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    #(≈ 3 lines of code)
    # n_x = ... 
    # n_h = ...
    # n_y = ... 
    # YOUR CODE STARTS HERE
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    
    # YOUR CODE ENDS HERE
    return (n_x, n_h, n_y)

     # 获取layer_sizes的测试案例， 这个案例中的数据和之前的那个无关
     t_X, t_Y = layer_sizes_test_case()

     # 计算层的大小
     (n_x, n_h, n_y) = layer_sizes(t_X, t_Y)

     # 打印输入层的大小
     print("输入层的大小为：n_x = " + str(n_x))

     # 打印隐藏层的大小
     print("隐藏层的大小为：n_h = " + str(n_h))

     # 打印输出层的大小
     print("输出层的大小为：n_y = " + str(n_y))

     # 运行layer_sizes的测试
     layer_sizes_test(layer_sizes)

![17](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/6fb1cf5e-6790-44bb-8c29-6a7e1d9f8b46)


### 在确定完shallow神经网络的structure之后，就是对数据进行initializing。

一般使用随机还原法，np.random.randn(a,b) * 0.01，不适用归0法是因为，如果hidden layer都是归0则，隐藏层没意义了.
<a name='4-2'></a>
### 4.2 - Initialize the model's parameters ####

<a name='ex-3'></a>
### Exercise 3 -  initialize_parameters

Implement the function `initialize_parameters()`.

**Instructions**:
- Make sure your parameters' sizes are right. Refer to the neural network figure above if needed.
- You will initialize the weights matrices with random values. 
    - Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
- You will initialize the bias vectors as zeros. 
    - Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.
 
    - 


     

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
 
**代码解释**


W1 = np.random.randn(n_h, n_x) * 0.01 
**其中，n_h是这个array的行，表示的是特征，所以表示这个layer神经元的数量，而n_x表示的是列，也就是上一个层级input到这个层级的数量**
  
b1 = np.zeros((n_h,1))
**b是一个常数列，他是不受上一层级影像的，所以后面是1**
W2 = np.random.randn(n_y, n_h) * 0.01
 b2 = np.zeros((n_y,1))


        # GRADED FUNCTION: initialize_parameters

        def initialize_parameters(n_x, n_h, n_y):
            """
            Argument:
            n_x -- size of the input layer
            n_h -- size of the hidden layer
            n_y -- size of the output layer
    
            Returns:
            params -- python dictionary containing your parameters:
                            W1 -- weight matrix of shape (n_h, n_x)
                            b1 -- bias vector of shape (n_h, 1)
                            W2 -- weight matrix of shape (n_y, n_h)
                            b2 -- bias vector of shape (n_y, 1)
            """    
            #(≈ 4 lines of code)
            # W1 = ...
            # b1 = ...
            # W2 = ...
            # b2 = ...
            # YOUR CODE STARTS HERE
            W1 = np.random.randn(n_h, n_x) * 0.01
            b1 = np.zeros((n_h,1))
            W2 = np.random.randn(n_y, n_h) * 0.01
            b2 = np.zeros((n_y,1))
    
            # YOUR CODE ENDS HERE

            parameters = {"W1": W1,
                          "b1": b1,
                          "W2": W2,
                          "b2": b2}
    
            return parameters

     
![18](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/12dba306-5c22-457e-830b-fe96df0940b0)


在定义完层级，
定义完initailize函数之后，就可以定义forward propagate 函数了.

<a name='4-3'></a>
### 4.3 - The Loop 

<a name='ex-4'></a>
### Exercise 4 - forward_propagation

Implement `forward_propagation()` using the following equations:

$$Z^{[1]} =  W^{[1]} X + b^{[1]}\tag{1}$$ 
$$A^{[1]} = \tanh(Z^{[1]})\tag{2}$$
$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}\tag{3}$$
$$\hat{Y} = A^{[2]} = \sigma(Z^{[2]})\tag{4}$$


**Instructions**:

- Check the mathematical representation of your classifier in the figure above.
- Use the function `sigmoid()`. It's built into (imported) this notebook.
- Use the function `np.tanh()`. It's part of the numpy library.
- Implement using these steps:
    1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()` by using `parameters[".."]`.
    2. Implement Forward Propagation. Compute $Z^{[1]}, A^{[1]}, Z^{[2]}$ and $A^{[2]}$ (the vector of all your predictions on all the examples in the training set).
- Values needed in the backpropagation are stored in "cache". The cache will be given as an input to the backpropagation function.

这次的hidden layer中使用的是，numpy library中的np.tanh()函数，并且tanh as the active function for hidden layer.


**Coding**

        # GRADED FUNCTION:forward_propagation

        def forward_propagation(X, parameters):
            """
            Argument:
            X -- input data of size (n_x, m)
            parameters -- python dictionary containing your parameters (output of initialization function)
    
            Returns:
            A2 -- The sigmoid output of the second activation
            cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
            """
            # Retrieve each parameter from the dictionary "parameters"
            #(≈ 4 lines of code)
            # W1 = ...
            # b1 = ...
            # W2 = ...
            # b2 = ...
            # YOUR CODE STARTS HERE
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
    
            # YOUR CODE ENDS HERE
    
            # Implement Forward Propagation to calculate A2 (probabilities)
            # (≈ 4 lines of code)
            # Z1 = ...
            # A1 = ...
            # Z2 = ...
            # A2 = ...
            # YOUR CODE STARTS HERE
            Z1 = np.dot(W1,X) + b1
            A1 = np.tanh(Z1)
            Z2 = np.dot(W2,A1) + b2
            A2 = 1/(1+np.exp(-Z2))
    
            # YOUR CODE ENDS HERE
    
            assert(A2.shape == (1, X.shape[1]))
    
            cache = {"Z1": Z1,
                     "A1": A1,
                     "Z2": Z2,
                     "A2": A2}
    
            return A2, cache

**以下是从测试数集中提取的**

![19](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/11dbc883-2b95-4071-85f0-751a2dec57d9)

<a name='4-4'></a>
### 4.4 - Compute the Cost

Now that you've computed $A^{[2]}$ (in the Python variable "`A2`"), which contains $a^{[2](i)}$ for all examples, you can compute the cost function as follows:

$$J = - \frac{1}{m} \sum\limits_{i = 1}^{m} \large{(} \small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right) \large{)} \small\tag{13}$$

<a name='ex-5'></a>
### Exercise 5 - compute_cost 

Implement `compute_cost()` to compute the value of the cost $J$.

**Instructions**:
- There are many ways to implement the cross-entropy loss. This is one way to implement one part of the equation without for loops:
$- \sum\limits_{i=1}^{m}  y^{(i)}\log(a^{[2](i)})$:
```python
logprobs = np.multiply(np.log(A2),Y)
cost = - np.sum(logprobs)          
```

- Use that to build the whole expression of the cost function.

**Notes**: 

- You can use either `np.multiply()` and then `np.sum()` or directly `np.dot()`).  
- If you use `np.multiply` followed by `np.sum` the end result will be a type `float`, whereas if you use `np.dot`, the result will be a 2D numpy array.  
- You can use `np.squeeze()` to remove redundant dimensions (in the case of single float, this will be reduced to a zero-dimension array). 
- You can also cast the array as a type `float` using `float()`.



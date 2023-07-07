### 构建一个2层hidden的神经网络和一个L神经网络.
**在多层神经网络构建时候，最方便的方法**

- 先定义一个能处理l层paramerters的initialization的function
- 再定义一个线性的forward方程
- 然后再根据需要的激活函数来构建联合方程（如果是sigmoid--relu）就再里面添加逻辑判断，activation == ？，然后来套刚才的forward方程=Z
  然后再A = g(Z)来保证不同的激活函数的工作.
- 通过 forward 中的cache（包含Z,W,b）可以计算cost.
- 然后进入backward的部分，同样先定义backward的线性方程
  ``` Python
  dW = 1/m*np.dot(dZ,A_prev.T)
  db = 1/m*np.sum(dZ,axis = 1, keep.dims=True)#记得是横向求和，并且保留dimension.
  dA_prev = np.dot(W.T,dZ)
  #而关于dZ的求值，不同的激活函数，有不同的结果
  #如果激活函数是softmax 和 sigmoid
  dZ = A - Y
  #如果激活函数是tanh和Relu
  dZ = dA * relu_derivative(Z)
 #其中，dA 是当前层的激活值的导数，relu_derivative 是 ReLU/tanh 函数的导数。
 ```
- 在设定好linear_backward函数后，我们需要根据用到的激活函数来设置前置激活函数：
 ``` Python
#sigmoid_backward（）
def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single sigmoid unit.

    Arguments:
    dA -- post-activation gradient, same shape as A
    cache -- 'Z' stored during forward propagation

    Returns:
    dZ -- gradient of the cost with respect to Z
    """
    Z = cache
    A = 1 / (1 + np.exp(-Z))
    dZ = dA * A * (1 - A)

    return dZ
#relu_backward
def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single ReLU unit.

    Arguments:
    dA -- post-activation gradient, same shape as A
    cache -- 'Z' stored during forward propagation

    Returns:
    dZ -- gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # 转换为与dA相同形状的数组

    # 当Z小于等于0时，将dZ置为0
    dZ[Z <= 0] = 0

    return dZ
#tanh_backward()
def tanh_backward(dA, cache):
    """
    Implement the backward propagation for a single tanh unit.

    Arguments:
    dA -- post-activation gradient, same shape as A
    cache -- 'A' stored during forward propagation

    Returns:
    dZ -- gradient of the cost with respect to Z
    """
    A = cache
    dZ = dA * (1 - np.power(A, 2))

    return dZ
```
- 然后根据sigmoid_backward算出的dZ值，带入linear_back的公式（dZ,activation_cache）算出dW，db,dA_prev.
<a name='1'></a>
## 1 - Packages

First, import all the packages you'll need during this assignment. 

- [numpy](www.numpy.org) is the main package for scientific computing with Python.
- [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
- dnn_utils provides some necessary functions for this notebook.
- testCases provides some test cases to assess the correctness of your functions
- np.random.seed(1) is used to keep all the random function calls consistent. It helps grade your work. Please don't change the seed!
``` Python
import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from public_tests import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)
```

<a name='2'></a>
## 2 - Outline

To build your neural network, you'll be implementing several "helper functions." These helper functions will be used in the next assignment to build a two-layer neural network and an L-layer neural network. 

Each small helper function will have detailed instructions to walk you through the necessary steps. Here's an outline of the steps in this assignment:

- Initialize the parameters for a two-layer network and for an $L$-layer neural network
- Implement the forward propagation module (shown in purple in the figure below)
     - Complete the LINEAR part of a layer's forward propagation step (resulting in $Z^{[l]}$).
     - The ACTIVATION function is provided for you (relu/sigmoid)
     - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
     - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). This gives you a new L_model_forward function.
- Compute the loss
- Implement the backward propagation module (denoted in red in the figure below)
    - Complete the LINEAR part of a layer's backward propagation step
    - The gradient of the ACTIVATION function is provided for you(relu_backward/sigmoid_backward) 
    - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function
    - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
- Finally, update the parameters

为了构建你的神经网络，你将会实现几个 "辅助函数"。这些辅助函数将在接下来的作业中被用来构建一个两层神经网络和一个L层神经网络。

每个小的辅助函数都会有详细的说明来指导你完成必要的步骤。下面是这项作业的步骤概要：

- 初始化两层网络和$L$层神经网络的参数
- 实现前向传播模块（下图中紫色显示）。
     - 完成一个层的前向传播步骤的线性部分（结果为$Z^{[l]}$）。
     - 为你提供了ACTIVATION函数（relu/sigmoid）。
     - 将前面两个步骤合并为一个新的[LINEAR->ACTIVATION]前向函数。
     - 将[LINEAR->RELU]正向函数堆叠L-1次（用于第1层到第L-1层），并在最后添加一个[LINEAR->SIGMOID]（用于最终层$L$）。这样你就得到了一个新的L_model_forward函数。
- 计算损失
- 实现后向传播模块(下图中用红色表示)
    - 完成一个层的后向传播步骤的LINEAR部分
    - 为你提供ACTIVATION函数的梯度(relu_backward/sigmoid_backward) 
    - 将前面两个步骤合并为一个新的[LINEAR->ACTIVATION]后向函数
    - 将[LINEAR->RELU]向后堆叠L-1次，并在一个新的L_model_backward函数中加入[LINEAR->SIGMOID]向后。
- 最后，更新参数

![31](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/03f77573-0c8c-4ce3-b75c-57fb89397fc3)

**Note**:

For every forward function, there is a corresponding backward function. This is why at every step of your forward module you will be storing some values in a cache. These cached values are useful for computing gradients. 

In the backpropagation module, you can then use the cache to calculate the gradients. Don't worry, this assignment will show you exactly how to carry out each of these steps! 

**注意**：

对于每个前向函数，都有一个相应的后向函数。这就是为什么在正向模块的每一步，你都要在缓存中存储一些数值。这些缓存的值对计算梯度很有用。

在反向传播模块中，你就可以使用缓存来计算梯度。别担心，本作业将向你展示如何进行这些步骤的具体操作! 

<a name='3'></a>
## 3 - Initialization

You will write two helper functions to initialize the parameters for your model. The first function will be used to initialize parameters for a two layer model. The second one generalizes this initialization process to $L$ layers.

<a name='3-1'></a>
### 3.1 - 2-layer Neural Network

<a name='ex-1'></a>
### Exercise 1 - initialize_parameters

Create and initialize the parameters of the 2-layer neural network.

**Instructions**:

- The model's structure is: *LINEAR -> RELU -> LINEAR -> SIGMOID*. 
- Use this random initialization for the weight matrices: `np.random.randn(d0, d1, ..., dn) * 0.01` with the correct shape. The documentation for [np.random.randn](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html)
- Use zero initialization for the biases: `np.zeros(shape)`. The documentation for [np.zeros](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)


3 - 初始化
你将写两个辅助函数来初始化你的模型的参数。第一个函数将用于初始化两层模型的参数。第二个函数将这个初始化过程推广到𝐿层。


3.1 - 2层神经网络

练习1 - 初始化_参数
创建并初始化2层神经网络的参数。

说明：

该模型的结构是： 线性->Rellu->线性->Sigmoid。
对权重矩阵使用这个随机初始化：np.random.randn(d0, d1, ..., dn) * 0.01，形状正确。np.random.randn的文档
对偏差使用零初始化：np.zeros(shape)。np.zeros的文档

**根据paramertes在shallow network中和layer shape的关系来撰写initialization function**
![32](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/42bfc2f8-b1f8-4691-bb58-680c308fa1af)

**CDOE**
# GRADED FUNCTION: initialize_parameters

``` Python
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    #(≈ 4 lines of code)
    # W1 = ...
    # b1 = ...
    # W2 = ...
    # b2 = ...
    # YOUR CODE STARTS HERE
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    
    # YOUR CODE ENDS HERE
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
```

![33](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/8979a802-0801-4a73-853b-056a4764c5f6)


      
<a name='3-2'></a>
### 3.2 - L-layer Neural Network

The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors. When completing the `initialize_parameters_deep` function, you should make sure that your dimensions match between each layer. Recall that $n^{[l]}$ is the number of units in layer $l$. For example, if the size of your input $X$ is $(12288, 209)$ (with $m=209$ examples) then:

更深的L层神经网络的初始化更为复杂，因为有更多的权重矩阵和偏置向量。在完成 initialize_parameters_deep 函数时，
你应该确保你的尺寸在每一层之间都是匹配的。回顾一下，𝑛[𝑙]是层𝑙的单位数。例如，如果你的输入𝑋的尺寸是（12288,209）（以𝑚=209为例），那么：

<table style="width:100%">
    <tr>
        <td>  </td> 
        <td> <b>Shape of W</b> </td> 
        <td> <b>Shape of b</b>  </td> 
        <td> <b>Activation</b> </td>
        <td> <b>Shape of Activation</b> </td> 
    <tr>
    <tr>
        <td> <b>Layer 1</b> </td> 
        <td> $(n^{[1]},12288)$ </td> 
        <td> $(n^{[1]},1)$ </td> 
        <td> $Z^{[1]} = W^{[1]}  X + b^{[1]} $ </td> 
        <td> $(n^{[1]},209)$ </td> 
    <tr>
    <tr>
        <td> <b>Layer 2</b> </td> 
        <td> $(n^{[2]}, n^{[1]})$  </td> 
        <td> $(n^{[2]},1)$ </td> 
        <td>$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$ </td> 
        <td> $(n^{[2]}, 209)$ </td> 
    <tr>
       <tr>
        <td> $\vdots$ </td> 
        <td> $\vdots$  </td> 
        <td> $\vdots$  </td> 
        <td> $\vdots$</td> 
        <td> $\vdots$  </td> 
    <tr>  
   <tr>
       <td> <b>Layer L-1</b> </td> 
        <td> $(n^{[L-1]}, n^{[L-2]})$ </td> 
        <td> $(n^{[L-1]}, 1)$  </td> 
        <td>$Z^{[L-1]} =  W^{[L-1]} A^{[L-2]} + b^{[L-1]}$ </td> 
        <td> $(n^{[L-1]}, 209)$ </td> 
   <tr>
   <tr>
       <td> <b>Layer L</b> </td> 
        <td> $(n^{[L]}, n^{[L-1]})$ </td> 
        <td> $(n^{[L]}, 1)$ </td>
        <td> $Z^{[L]} =  W^{[L]} A^{[L-1]} + b^{[L]}$</td>
        <td> $(n^{[L]}, 209)$  </td> 
    <tr>
</table>

Remember that when you compute $W X + b$ in python, it carries out broadcasting. For example, if: 

$$ W = \begin{bmatrix}
    w_{00}  & w_{01} & w_{02} \\
    w_{10}  & w_{11} & w_{12} \\
    w_{20}  & w_{21} & w_{22} 
\end{bmatrix}\;\;\; X = \begin{bmatrix}
    x_{00}  & x_{01} & x_{02} \\
    x_{10}  & x_{11} & x_{12} \\
    x_{20}  & x_{21} & x_{22} 
\end{bmatrix} \;\;\; b =\begin{bmatrix}
    b_0  \\
    b_1  \\
    b_2
\end{bmatrix}\tag{2}$$

Then $WX + b$ will be:

$$ WX + b = \begin{bmatrix}
    (w_{00}x_{00} + w_{01}x_{10} + w_{02}x_{20}) + b_0 & (w_{00}x_{01} + w_{01}x_{11} + w_{02}x_{21}) + b_0 & \cdots \\
    (w_{10}x_{00} + w_{11}x_{10} + w_{12}x_{20}) + b_1 & (w_{10}x_{01} + w_{11}x_{11} + w_{12}x_{21}) + b_1 & \cdots \\
    (w_{20}x_{00} + w_{21}x_{10} + w_{22}x_{20}) + b_2 &  (w_{20}x_{01} + w_{21}x_{11} + w_{22}x_{21}) + b_2 & \cdots
\end{bmatrix}\tag{3}  $$


<a name='ex-2'></a>
### Exercise 2 -  initialize_parameters_deep

Implement initialization for an L-layer Neural Network. 

**Instructions**:
- The model's structure is *[LINEAR -> RELU] $ \times$ (L-1) -> LINEAR -> SIGMOID*. I.e., it has $L-1$ layers using a ReLU activation function followed by an output layer with a sigmoid activation function.
- Use random initialization for the weight matrices. Use `np.random.randn(d0, d1, ..., dn) * 0.01`.
- Use zeros initialization for the biases. Use `np.zeros(shape)`.
- You'll store $n^{[l]}$, the number of units in different layers, in a variable `layer_dims`. For example, the `layer_dims` for last week's Planar Data classification model would have been [2,4,1]: There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit. This means `W1`'s shape was (4,2), `b1` was (4,1), `W2` was (1,4) and `b2` was (1,1). Now you will generalize this to $L$ layers! 
- Here is the implementation for $L=1$ (one layer neural network). It should inspire you to implement the general case (L-layer neural network).

  练习 2 - initialize_parameters_deep
实现L层神经网络的初始化。

指示：

该模型的结构是*[LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID*。也就是说，它有𝐿-1个使用ReLU激活函数的层，然后是一个使用sigmoid激活函数的输出层。
对权重矩阵使用随机初始化。使用np.random.randn(d0, d1, ..., dn) * 0.01。
对偏置使用零的初始化。使用 np.zeros(shape)。
你将把𝑛[𝑙]，不同层的单元数，存储在变量 layer_dims 中。例如，上周的平面数据分类模型的 layer_dims 应该是 [2,4,1]： 有两个输入，一个有4个隐藏单元的隐藏层，以及一个有1个输出单元的输出层。这意味着W1的形状是（4,2），b1是（4,1），W2是（1,4），b2是（1,1）。现在，你将把它推广到𝐿层!
下面是对𝐿=的实现。

```python
    if L == 1:
        parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
        parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))
```
**也是相同的，我们需要按照W,b这两个paramertes的位置来写initiation的代码**
而由于是有多层layer，我们需要使用for loop函数来帮助我们。
首先把shape of layer储存到vertor：layer_dims中，
使用for  l in range(1, L):

**COding**

```python
# GRADED FUNCTION: initialize_parameters_deep

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        #(≈ 2 lines of code)
        # parameters['W' + str(l)] = ...
        # parameters['b' + str(l)] = ...
        # YOUR CODE STARTS HERE
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        # YOUR CODE ENDS HERE
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters
```
![34](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/457dcea7-d819-48dc-a175-066a813d8265)

<a name='4'></a>
## 4 - Forward Propagation Module

<a name='4-1'></a>
### 4.1 - Linear Forward 

Now that you have initialized your parameters, you can do the forward propagation module. Start by implementing some basic functions that you can use again later when implementing the model. Now, you'll complete three functions in this order:

- LINEAR
- LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid. 
- [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID (whole model)

The linear forward module (vectorized over all the examples) computes the following equations:

4 - 前向传播模块

4.1 - 线性前向
现在你已经初始化了你的参数，你可以做前向传播模块了。从实现一些基本函数开始，你可以在以后实现模型时再次使用。现在，你将按照这个顺序完成三个函数：

LINEAR
LINEAR -> ACTIVATION，其中ACTIVATION将是ReLU或Sigmoid。
[LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID (整个模型)
线性前向模块（对所有的例子进行矢量计算）计算出以下方程：

$$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}\tag{4}$$

where $A^{[0]} = X$. 

<a name='ex-3'></a>
### Exercise 3 - linear_forward 

Build the linear part of forward propagation.

**Reminder**:
The mathematical representation of this unit is $Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$. You may also find `np.dot()` useful. If your dimensions don't match, printing `W.shape` may help.

练习3 - linear_forward
建立前向传播的线性部分。

提醒一下： 这个单元的数学表示是：𝑍[𝑙]=𝑊[𝑙]𝐴[𝑙-1]+𝑏[𝑙] 。你可能还会发现np.dot()很有用。如果你的尺寸不匹配，打印W.shape可能有帮助。


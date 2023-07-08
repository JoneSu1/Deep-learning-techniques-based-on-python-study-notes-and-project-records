### 构建一个2层hidden的神经网络和一个L神经网络.
**在多层神经网络构建时候，最方便的方法**

- 先定义一个能处理l层paramerters的initialization的function
- 再定义一个线性的forward方程
- 然后再根据需要的激活函数来构建联合方程（如果是sigmoid--relu）就再里面添加逻辑判断，activation == ？，然后来套刚才的forward方程=Z
  然后再A = g(Z)来保证不同的激活函数的工作.
  同样，需要先定义sigmoid和relu的helper函数，由于这个L神经网络的组成是L-1个relu，1个sigmoid的output.
  ``` Python
  #当激活函数是sigmoid时候
  sigmoid = 1/(1+np.exp(-Z))
  #当激活函数是relu的时候
  relu = np.maximum(0,Z)
  reture Z
  ```
- 通过 forward 中的cache（包含Z,W,b）可以计算cost.
  而costfunction是： $$-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right)) \tag{7}$$
  其中Y*log（AL）因为都是array，想元素成元素，就得使用np.multiply（）函数.
         cost = -1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
    得到cost 值之后，就可以进入backward propagation
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

- 下一步就是计算出dZ值，以及定义出gradient descent function.
  其中需要记住的是：dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
**以及，在L=number of layer时候，A只有L-1个，W有L个，b也有L个**
  然后因为是backward propagation所以是从大的layer倒着算gra的
   for l in reversed(range(L-1))，所以我们需要使用reversed（range（））这个函数，帮我们取l值，从L-1开始取.
  然后这是L层nn的back_ward propagation函数设置。
``` python
      for l in reversed(range(L-1)): #这里的l in reversed(range(L-1)),其中L是layer数，而L-1是为了去掉input层
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        #(approx. 5 lines)
        # current_cache = ...
        # dA_prev_temp, dW_temp, db_temp = ...
        # grads["dA" + str(l)] = ...
        # grads["dW" + str(l + 1)] = ...
        # grads["db" + str(l + 1)] = ...
        # YOUR CODE STARTS HERE
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp #这里是l而不是l+1是因为 dA是前一个layer里面的A的导数.
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        # YOUR CODE ENDS HERE
```
- 在得到  dZ值和gradient function之后，就要用for loop写出能更新每一个layer参数的代码：并定义成update_parameters（params, grads, learning_rate）function
  ```Python
   parameters = params.copy()#
   L = len(parameters) // 2 # number of layers in the neural network,因为parameters里面有W,b。他们的数量是layers的两倍.
   parameters["W" + str(l+1)] = params["W"+ str(l+1)] - learning_rate * grads["dW"+ str(l+1)]# l+1是为了避开input那一层
   parameters["b" + str(l+1)] = params["b"+ str(l+1)] - learning_rate * grads["db"+ str(l+1)]
  ```
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

```Python
# GRADED FUNCTION: linear_forward

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    #(≈ 1 line of code)
    # Z = ...
    # YOUR CODE STARTS HERE
    Z = np.dot(W,A) + b
    
    # YOUR CODE ENDS HERE
    cache = (A, W, b)
    
    return Z, cache
```
![35](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/0a5c99c2-83cf-497c-bac9-aedf2903c067)


<a name='4-2'></a>
### 4.2 - Linear-Activation Forward
**def linear_forward(A, W, b):**
关于如何构sigmoid（Z）和relu（Z）公式在上面
In this notebook, you will use two activation functions:

- **Sigmoid**: $\sigma(Z) = \sigma(W A + b) = \frac{1}{ 1 + e^{-(W A + b)}}$. You've been provided with the `sigmoid` function which returns **two** items: the activation value "`a`" and a "`cache`" that contains "`Z`" (it's what we will feed in to the corresponding backward function). To use it you could just call: 
``` python
A, activation_cache = sigmoid(Z)
```

- **ReLU**: The mathematical formula for ReLu is $A = RELU(Z) = max(0, Z)$. You've been provided with the `relu` function. This function returns **two** items: the activation value "`A`" and a "`cache`" that contains "`Z`" (it's what you'll feed in to the corresponding backward function). To use it you could just call:
``` python
A, activation_cache = relu(Z)
```

For added convenience, you're going to group two functions (Linear and Activation) into one function (LINEAR->ACTIVATION). Hence, you'll implement a function that does the LINEAR forward step, followed by an ACTIVATION forward step.

<a name='ex-4'></a>
### Exercise 4 - linear_activation_forward

Implement the forward propagation of the *LINEAR->ACTIVATION* layer. Mathematical relation is: $A^{[l]} = g(Z^{[l]}) = g(W^{[l]}A^{[l-1]} +b^{[l]})$ where the activation "g" can be sigmoid() or relu(). Use `linear_forward()` and the correct activation function.

为了方便起见，你要把两个函数（线性和激活）组合成一个函数（线性->激活）。因此，你要实现一个函数，先做线性前进步骤，然后再做激活前进步骤。


练习4 - 线性激活_前向
实现LINEAR->ACTIVATION层的前向传播。数学关系是：𝐴[𝑙]=𝑔(𝑍[𝑙])=𝑔(𝑊[𝑙]𝐴[𝑙-1]+𝑏[𝑙]) 其中激活的g可以是sigmoid（）或是relu（）。使用 linear_forward() 和正确的激活函数。

**linear_activation_forward(A_prev, W, b, activation):**
``` Python
# GRADED FUNCTION: linear_activation_forward

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        #(≈ 2 lines of code)
        # Z, linear_cache = ...
        # A, activation_cache = ...
        # YOUR CODE STARTS HERE
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
        
        # YOUR CODE ENDS HERE
    
    elif activation == "relu":
        #(≈ 2 lines of code)
        # Z, linear_cache = ...
        # A, activation_cache = ...
        # YOUR CODE STARTS HERE
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
        
        # YOUR CODE ENDS HERE
    cache = (linear_cache, activation_cache)

    return A, cache
```
![36](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/3f43dd6d-0c12-46e5-9736-cbb94d7bd9cd)

注意：在深度学习中，"[LINEAR->ACTIVATION]"的计算被算作神经网络的单层，而不是两层。

**在定义完了forward的激活函数之后，就可以通过for loop函数定义出forward的L modle了**
<a name='4-3'></a>
### 4.3 - L-Layer Model 
**L_model_forward(X, parameters):**
For even *more* convenience when implementing the $L$-layer Neural Net, you will need a function that replicates the previous one (`linear_activation_forward` with RELU) $L-1$ times, then follows that with one `linear_activation_forward` with SIGMOID.
4.3 - L层模型

为了在实现𝐿层神经网络时更加方便，你需要一个函数来复制前一个函数（带RELU的线性激活_前向）𝐿-1次，然后再用一个带SIGMOID的线性激活_前向函数。 
![37](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/80806f63-27b5-41ea-b1b7-9990794c17c4)

<a name='ex-5'></a>
### Exercise 5 -  L_model_forward

Implement the forward propagation of the above model.

**Instructions**: In the code below, the variable `AL` will denote $A^{[L]} = \sigma(Z^{[L]}) = \sigma(W^{[L]} A^{[L-1]} + b^{[L]})$. (This is sometimes also called `Yhat`, i.e., this is $\hat{Y}$.) 

**Hints**:
- Use the functions you've previously written 
- Use a for loop to replicate [LINEAR->RELU] (L-1) times
  
练习5 - L_model_forward
实现上述模型的向前传播。

说明： 在下面的代码中，变量AL表示𝐴[𝐿]=𝜎(𝑍[𝐿])=𝜎(𝑊[𝐿]𝐴[𝐿-1]+𝑏[𝐿] ) 。(这有时也被称为Yhat，即这是𝑌̂。）

提示：

- 使用你以前写过的函数
- 使用for循环来复制[LINEAR->RELU]（L-1）次
- 不要忘记跟踪 "缓存 "列表中的缓存。要在列表中添加一个新的值c，你可以使用list.append(c)。
- Don't forget to keep track of the caches in the "caches" list. To add a new value `c` to a `list`, you can use `list.append(c)`.
  
它接收了输入X，并输出了一个包含你的预测的行向量𝐴[𝐿]！你实现了一个完全的前向传播，它接受输入X并输出一个包含你的预测的行向量𝐴[𝐿]。
它还在 "缓存 "中记录了所有的中间值。使用𝐴[𝐿]，你可以计算你的预测的成本。
```Python
# GRADED FUNCTION: L_model_forward

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A 
        #(≈ 2 lines of code)
        # A, cache = ...
        # caches ...
        # YOUR CODE STARTS HERE
        
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
        caches.append(cache)
        # YOUR CODE ENDS HERE
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    #(≈ 2 lines of code)
    # AL, cache = ...
    # caches ...
    # YOUR CODE STARTS HERE
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)
    # YOUR CODE ENDS HERE
          
    return AL, caches
```
![38](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/0c1abbde-90f5-4910-a276-612627390908)

**已经定义出了可以计算每一层的激活函数值的函数L_modle_forward(X,parameters)**
下一步就可以计算出forward propagation的cost函数了.
<a name='5'></a>
## 5 - Cost Function
**compute_cost（AL,Y）**

Now you can implement forward and backward propagation! You need to compute the cost, in order to check whether your model is actually learning.

<a name='ex-6'></a>
### Exercise 6 - compute_cost
Compute the cross-entropy cost $J$, using the following formula: 
![44](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a5fe96a5-5faf-45c6-a82c-d4712b3ccb60)


5 - 成本函数
现在你可以实现前向和后向传播了！你需要计算成本！你需要计算成本，以便检查你的模型是否真的在学习。


练习6 - 计算成本（compute_cost
计算交叉熵成本𝐽 ，使用以下公式：

# GRADED FUNCTION: compute_cost
```Python
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    # (≈ 1 lines of code)
    # cost = ...
    # YOUR CODE STARTS HERE
    
    cost = -1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
    # YOUR CODE ENDS HERE
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    
    return cost
```
![40](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/00dda6af-340b-46df-a2ef-36828cef57b3)

**现在前向的参数已经都存到cache中了，可以计算backward的了**
**同样也是定义backward的函数，求dZ,dW,db,dAprev**
<a name='6'></a>
## 6 - Backward Propagation Module

Just as you did for the forward propagation, you'll implement helper functions for backpropagation. Remember that backpropagation is used to calculate the gradient of the loss function with respect to the parameters. 

6 - 后向传播模块
就像你为正向传播所做的那样，你将为反向传播实现辅助函数。记住，反向传播是用来计算损失函数相对于参数的梯度的。

![41](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a8dc8786-d63f-497a-862e-bd5555af3514)


**Reminder**: 

<!-- 
For those of you who are experts in calculus (which you don't need to be to do this assignment!), the chain rule of calculus can be used to derive the derivative of the loss $\mathcal{L}$ with respect to $z^{[1]}$ in a 2-layer network as follows:

$$\frac{d \mathcal{L}(a^{[2]},y)}{{dz^{[1]}}} = \frac{d\mathcal{L}(a^{[2]},y)}{{da^{[2]}}}\frac{{da^{[2]}}}{{dz^{[2]}}}\frac{{dz^{[2]}}}{{da^{[1]}}}\frac{{da^{[1]}}}{{dz^{[1]}}} \tag{8} $$

In order to calculate the gradient $dW^{[1]} = \frac{\partial L}{\partial W^{[1]}}$, use the previous chain rule and you do $dW^{[1]} = dz^{[1]} \times \frac{\partial z^{[1]} }{\partial W^{[1]}}$. During backpropagation, at each step you multiply your current gradient by the gradient corresponding to the specific layer to get the gradient you wanted.

Equivalently, in order to calculate the gradient $db^{[1]} = \frac{\partial L}{\partial b^{[1]}}$, you use the previous chain rule and you do $db^{[1]} = dz^{[1]} \times \frac{\partial z^{[1]} }{\partial b^{[1]}}$.

This is why we talk about **backpropagation**.
!-->

Now, similarly to forward propagation, you're going to build the backward propagation in three steps:
1. LINEAR backward
2. LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation
3. [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID backward (whole model)


现在，与前向传播类似，你要分三步建立后向传播：

- 向后的LINEAR
- LINEAR -> ACTIVATION向后，其中ACTIVATION计算ReLU或sigmoid激活的导数
- [LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID 向后（整个模型）。
 
在接下来的练习中，你需要记住：：

b是一个1列n行的矩阵(np.ndarray)，即：b = [[1.0], [2.0]] (记住b是一个常数)
np.sum对ndarray的元素进行求和。
axis=1或axis=0分别指定是按行还是按列进行求和
keepdims指定是否必须保留矩阵的原始尺寸。

<a name='6-1'></a>
### 6.1 - Linear Backward
**linear_backward(dZ, cache)**
For layer $l$, the linear part is: $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ (followed by an activation).

Suppose you have already calculated the derivative $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}}$. You want to get $(dW^{[l]}, db^{[l]}, dA^{[l-1]})$.
6.1 - 线性后退
对于层𝑙，线性部分是：  𝑍[𝑙]=𝑊[𝑙]𝐴[𝑙-1]+𝑏[𝑙]（后面是一个激活）。

假设你已经计算了导数𝑑𝑍[𝑙]=∂∂𝑍[𝑙] 。你想得到（𝑑𝑊[𝑙],𝑑𝑏[𝑙],𝑑𝐴[𝑙-1]） 。
![42](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/0d0d5f29-5bd6-4f52-b336-770d1b12e6ef)

The three outputs $(dW^{[l]}, db^{[l]}, dA^{[l-1]})$ are computed using the input $dZ^{[l]}$.
三个输出（𝑑𝑊[𝑙],𝑑𝑏[𝑙],𝑑𝐴[𝑙-1]）是使用输入𝑑𝑍[𝑙] 计算的。


Here are the formulas you need: 
![43](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a1397593-e378-47d5-afc6-660f8cc2be74)


练习7 - linear_backward
使用上面的3个公式来实现 linear_backward()。

提示：

在numpy中，你可以使用A.T或A.transpose()来获得一个ndarray A的转置。
```Python
# GRADED FUNCTION: linear_backward

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    # dW = ...
    # db = ... sum by the rows of dZ with keepdims=True
    # dA_prev = ...
    # YOUR CODE STARTS HERE
    
    dW = 1/m *np.dot(dZ,A_prev.T)
    db = 1/m * np.sum(dZ,axis = 1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    # YOUR CODE ENDS HERE
    
    return dA_prev, dW, db
```

 ![45](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/00095ca8-189c-4d30-bf57-25ff90d1ce6c)

 **定义完了backward function下一步就是定义激活函数了**
<a name='6-2'></a>
### 6.2 - Linear-Activation Backward
**linear_activation_backward(dA, cache, activation)**
关于如何构建出backward的helper函数在上面.
Next, you will create a function that merges the two helper functions: **`linear_backward`** and the backward step for the activation **`linear_activation_backward`**. 

To help you implement `linear_activation_backward`, two backward functions have been provided:
- **`sigmoid_backward`**: Implements the backward propagation for SIGMOID unit. You can call it as follows:

```python
dZ = sigmoid_backward(dA, activation_cache)
```

- **`relu_backward`**: Implements the backward propagation for RELU unit. You can call it as follows:

```python
dZ = relu_backward(dA, activation_cache)
```

If $g(.)$ is the activation function, 
`sigmoid_backward` and `relu_backward` compute $$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]}). \tag{11}$$  

<a name='ex-8'></a>
### Exercise 8 -  linear_activation_backward
Implement the backpropagation for the *LINEAR->ACTIVATION* layer.

```python
# GRADED FUNCTION: linear_activation_backward

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        #(≈ 2 lines of code)
        # dZ =  ...
        # dA_prev, dW, db =  ...
        # YOUR CODE STARTS HERE
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
        # YOUR CODE ENDS HERE
        
    elif activation == "sigmoid":
        #(≈ 2 lines of code)
        # dZ =  ...
        # dA_prev, dW, db =  ...
        # YOUR CODE STARTS HERE
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
        # YOUR CODE ENDS HERE
    
    return dA_prev, dW, db

```
![46](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/3b09bd6b-2143-4fe8-9104-d30f87ef9965)


<a name='6-3'></a>
### 6.3 - L-Model Backward 
**L_model_backward(AL, Y, caches）**
Now you will implement the backward function for the whole network! 

Recall that when you implemented the `L_model_forward` function, at each iteration, you stored a cache which contains (X,W,b, and z). In the back propagation module, you'll use those variables to compute the gradients. Therefore, in the `L_model_backward` function, you'll iterate through all the hidden layers backward, starting from layer $L$. On each step, you will use the cached values for layer $l$ to backpropagate through layer $l$. Figure 5 below shows the backward pass. 

6.3 - L-模型后退
现在你将实现整个网络的后向函数!

回想一下，当你实现L_model_forward函数时，在每次迭代时，你存储了一个包含(X,W,b,和z)的缓存。在反向传播模块中，你将使用这些变量来计算梯度。因此，在L_model_backward函数中，你将从𝐿层开始，向后迭代所有隐藏层。在每一步中，你将使用层𝑙的缓存值来反向传播层𝑙。下面的图5显示了后向传递。
![47](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/ad181ebc-fd26-4a45-ad73-e298e6e8d5aa)
![48](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d1c2cff4-27d8-4e10-b051-3866fb4ee1be)



初始化反向传播：

要通过这个网络进行反向传播，你知道，输出是：  𝐴[𝐿]=𝜎(𝑍[𝐿]) 。因此，你的代码需要计算dAL =∂∂𝐴[𝐿]。要做到这一点，请使用这个公式（使用微积分得出，同样，你不需要深入了解！）：

dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL)) # 成本相对于AL的导数
然后你可以使用这个激活后的梯度dAL来继续往后走。如图5所示，你现在可以将dAL送入你实现的LINEAR->SIGMOID向后函数中（它将使用L_model_forward函数存储的缓存值）。

之后，你将不得不使用for循环，用LINEAR->RELU后向函数遍历所有其他层。你应该将每个dA、dW和db存储在grads字典中。要做到这一点，请使用这个公式：

𝑔𝑟𝑎𝑑𝑠["𝑑𝑊"+𝑠𝑡𝑟(𝑙)]=𝑑𝑊[𝑙](15)
例如，对于𝑙=3，这将把𝑑𝑊[𝑙]存入grads["dW3"]。


练习9 - L_model_backward
对*[LINEAR->RELU] × (L-1) -> LINEAR -> SIGMOID*模型实施反向传播。

```python
# GRADED FUNCTION: L_model_backward

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    #(1 line of code)
    # dAL = ...
    # YOUR CODE STARTS HERE
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))    
    # YOUR CODE ENDS HERE
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    #(approx. 5 lines)
    # current_cache = ...
    # dA_prev_temp, dW_temp, db_temp = ...
    # grads["dA" + str(L-1)] = ...
    # grads["dW" + str(L)] = ...
    # grads["db" + str(L)] = ...
    # YOUR CODE STARTS HERE#这是定义的最后输出的sigmoid的参数
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    print("dA"+ str(L-1)+" = "+str(grads["dA" + str(L-1)]))
    print("dW"+ str(L)+" = "+str(grads["dW" + str(L)]))
    print("db"+ str(L)+" = "+str(grads["db" + str(L)]))
    # YOUR CODE ENDS HERE
    
    # Loop from l=L-2 to l=0 #这是在定义1到L-1的relu的参数
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        #(approx. 5 lines)
        # current_cache = ...
        # dA_prev_temp, dW_temp, db_temp = ...
        # grads["dA" + str(l)] = ...
        # grads["dW" + str(l + 1)] = ...
        # grads["db" + str(l + 1)] = ...
        # YOUR CODE STARTS HERE
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        # YOUR CODE ENDS HERE

    return grads
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/22ee5c81-49a7-4bf2-8b96-3f171a90a628).

**下一步就是更新parameters**

<a name='6-4'></a>
### 6.4 - Update Parameters
**update_parameters(params, grads, learning_rate):**
In this section, you'll update the parameters of the model, using gradient descent: 

$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{16}$$
$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{17}$$


![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/0182df52-5583-40fe-9b0f-d29ef1b26c99)

where $\alpha$ is the learning rate. 

After computing the updated parameters, store them in the parameters dictionary. 


Exercise 10 - update_parameters
Implement update_parameters() to update your parameters using gradient descent.

Instructions: Update parameters using gradient descent on every  𝑊[𝑙]  and  𝑏[𝑙]  for  𝑙=1,2,...,𝐿 .
```python
# GRADED FUNCTION: update_parameters

def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    params -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = params.copy()
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    #(≈ 2 lines of code)
    for l in range(L):
        # parameters["W" + str(l+1)] = ...
        # parameters["b" + str(l+1)] = ...
        # YOUR CODE STARTS HERE
        parameters["W" + str(l+1)] = params["W"+ str(l+1)] - learning_rate * grads["dW"+ str(l+1)]
        parameters["b" + str(l+1)] = params["b"+ str(l+1)] - learning_rate * grads["db"+ str(l+1)]
        # YOUR CODE ENDS HERE
    return parameters
```


### Congratulations! 

You've just implemented all the functions required for building a deep neural network, including: 

- Using non-linear units improve your model
- Building a deeper neural network (with more than 1 hidden layer)
- Implementing an easy-to-use neural network class

This was indeed a long assignment, but the next part of the assignment is easier. ;) 

In the next assignment, you'll be putting all these together to build two models:

- A two-layer neural network
- An L-layer neural network

You will in fact use these models to classify cat vs non-cat images! (Meow!) Great work and see you next time. 

你刚刚实现了构建一个深度神经网络所需的所有功能，包括：

- 使用非线性单元改善你的模型
- 构建一个更深的神经网络（有1个以上的隐藏层）
- 实现一个易于使用的神经网络类
- 这的确是一个很长的作业，但下一部分作业更容易。;)

在接下来的作业中，你将把所有这些放在一起，建立两个模型：

一个两层的神经网络
一个L层的神经网络
事实上，你们将使用这些模型来对猫和非猫的图像进行分类 (Meow!) 干得好，下次见。

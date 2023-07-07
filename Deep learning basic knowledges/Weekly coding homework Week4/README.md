### 构建一个2层hidden的神经网络和一个L神经网络.

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


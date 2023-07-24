# Convolutional Neural Networks: Step by Step

Welcome to Course 4's first assignment! In this assignment, you will implement convolutional (CONV) and pooling (POOL) layers in numpy, including both forward propagation and (optionally) backward propagation. 

By the end of this notebook, you'll be able to: 

* Explain the convolution operation
* Apply two different types of pooling operation
* Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
* Build a convolutional neural network
  
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/ba874069-5063-46d7-98e8-e9785ecad94f)


## Table of Contents

- [1 - Packages](#1)
- [2 - Outline of the Assignment](#2)
- [3 - Convolutional Neural Networks](#3)
    - [3.1 - Zero-Padding](#3-1)
        - [Exercise 1 - zero_pad](#ex-1)
    - [3.2 - Single Step of Convolution](#3-2)
        - [Exercise 2 - conv_single_step](#ex-2)
    - [3.3 - Convolutional Neural Networks - Forward Pass](#3-3)
        - [Exercise 3 - conv_forward](#ex-3)
- [4 - Pooling Layer](#4)
    - [4.1 - Forward Pooling](#4-1)
        - [Exercise 4 - pool_forward](#ex-4)
- [5 - Backpropagation in Convolutional Neural Networks (OPTIONAL / UNGRADED)](#5)
    - [5.1 - Convolutional Layer Backward Pass](#5-1)
        - [5.1.1 - Computing dA](#5-1-1)
        - [5.1.2 - Computing dW](#5-1-2)
        - [5.1.3 - Computing db](#5-1-3)
            - [Exercise 5 - conv_backward](#ex-5)
    - [5.2 Pooling Layer - Backward Pass](#5-2)
        - [5.2.1 Max Pooling - Backward Pass](#5-2-1)
            - [Exercise 6 - create_mask_from_window](#ex-6)
        - [5.2.2 - Average Pooling - Backward Pass](#5-2-2)
            - [Exercise 7 - distribute_value](#ex-7)
        - [5.2.3 Putting it Together: Pooling Backward](#5-2-3)
            - [Exercise 8 - pool_backward](#ex-8)
         
<a name='1'></a>
## 1 - Packages

Let's first import all the packages that you will need during this assignment. 
- [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.
- [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
- np.random.seed(1) is used to keep all the random function calls consistent. This helps to grade your work.
```python
import numpy as np
import h5py
import matplotlib.pyplot as plt
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
## 2 - Outline of the Assignment

You will be implementing the building blocks of a convolutional neural network! Each function you will implement will have detailed instructions to walk you through the steps:

- Convolution functions, including:
    - Zero Padding
    - Convolve window 
    - Convolution forward
    - Convolution backward (optional)
- Pooling functions, including:
    - Pooling forward
    - Create mask 
    - Distribute value
    - Pooling backward (optional)
    
This notebook will ask you to implement these functions from scratch in `numpy`. In the next notebook, you will use the TensorFlow equivalents of these functions to build the following model:

**Note**: For every forward function, there is a corresponding backward equivalent. Hence, at every step of your forward module you will store some parameters in a cache. These parameters are used to compute gradients during backpropagation. 

<a name='2'></a>
## 2 - 作业大纲

您将实现卷积神经网络的构建模块！你要实现的每个函数都有详细的说明，指导你完成各个步骤：

- 卷积函数，包括
    - 零填充
    - 卷积窗口 
    - 前向卷积
    - 后向卷积（可选）
- 池化功能，包括
    - 向前汇集
    - 创建掩码 
    - 分配值
    - 向后汇集（可选）
    
本笔记本将要求您在 `numpy` 中从头开始实现这些函数。在下一个笔记本中，您将使用这些函数的 TensorFlow 对应函数来构建以下模型：

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a10b6d4e-9444-4223-951c-c924666f605d)


**注意**： 每个前向函数都有一个对应的后向等价函数。因此，前向模块的每一步都会在缓存中存储一些参数。这些参数用于计算反向传播过程中的梯度。

<a name='3'></a>
## 3 - Convolutional Neural Networks

Although programming frameworks make convolutions easy to use, they remain one of the hardest concepts to understand in Deep Learning. A convolution layer transforms an input volume into an output volume of different size, as shown below. 



In this part, you will build every step of the convolution layer. You will first implement two helper functions: one for zero padding and the other for computing the convolution function itself.

<a name='3'></a>
## 3 - 卷积神经网络

虽然编程框架让卷积变得简单易用，但卷积仍然是深度学习中最难理解的概念之一。卷积层将输入体积转换为不同大小的输出体积，如下图所示。
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a795e9ca-1154-4881-a4a4-4714410712e0)


在这一部分中，您将构建卷积层的每一个步骤。首先，您将实现两个辅助函数：一个用于零填充，另一个用于计算卷积函数本身。

<a name='3-1'></a>
### 3.1 - Zero-Padding(0填充)

Zero-padding adds zeros around the border of an image:
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/fc970f07-71fa-4879-a31f-5abf00f17afa)

The main benefits of padding are:

- It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers. An important special case is the "same" convolution, in which the height/width is exactly preserved after one layer. 

- It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels at the edges of an image.

填充的主要优点有

- 它允许您使用 CONV 层，而不必缩小卷的高度和宽度。这对于构建更深的网络非常重要，否则高度/宽度会随着层数的增加而缩小。一个重要的特例是 "相同 "卷积，在这种情况下，高度/宽度在一层之后完全保留。

- 这有助于我们在图像边界保留更多信息。如果没有填充，下一层中很少有数值会受到图像边缘像素的影响。


<a name='ex-1'></a>
### Exercise 1 - zero_pad
Implement the following function, which pads all the images of a batch of examples X with zeros. [Use np.pad](https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html). Note if you want to pad the array "a" of shape $(5,5,5,5,5)$ with `pad = 1` for the 2nd dimension, `pad = 3` for the 4th dimension and `pad = 0` for the rest, you would do:
```python
a = np.pad(a, ((0,0), (1,1), (0,0), (3,3), (0,0)), mode='constant', constant_values = (0,0))
```

在这个函数中，我们要对一个四维的numpy数组X进行填充，这个数组的形状是(m, n_H, n_W, n_C)，其中：

m是图像的数量，
n_H是图像的高度，
n_W是图像的宽度，
n_C是颜色通道的数量。
我们要在图像的高度和宽度上添加填充，而不需要在图像的数量或颜色通道上添加填充。这就是为什么我们在np.pad函数的第二个参数中使用了((0, 0), (pad, pad), (pad, pad), (0, 0))。这个参数是一个元组，定义了在每个维度上需要添加的填充的数量。在每个元组中，第一个数字是在该维度的开始添加的填充的数量，第二个数字是在该维度的结束添加的填充的数量。

(0, 0)表示在第一个维度（图像的数量）上不添加填充，
(pad, pad)表示在第二个维度（图像的高度）上在开始和结束都添加pad数量的填充，
(pad, pad)表示在第三个维度（图像的宽度）上在开始和结束都添加pad数量的填充，
(0, 0)表示在第四个维度（颜色通道）上不添加填充。
mode='constant'表示我们要添加的是常数填充，constant_values=0表示我们添加的常数值是0。所以，我们在各个维度上添加的填充都是0。

# GRADED FUNCTION: zero_pad
```python
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    
    #(≈ 1 line)
    # X_pad = None
    # YOUR CODE STARTS HERE
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)), mode = "constant", constant_values = (0,0))
    # YOUR CODE ENDS HERE
    
    return X_pad
```
**Testing**
```python
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 3)
print ("x.shape =\n", x.shape)
print ("x_pad.shape =\n", x_pad.shape)
print ("x[1,1] =\n", x[1, 1])
print ("x_pad[1,1] =\n", x_pad[1, 1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])
zero_pad_test(zero_pad)
```
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/e43f6c84-5181-4513-853f-d5da6a08800b)
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/26dd772a-e096-4880-b956-ac6a5c6ab3fe)

<a name='3-2'></a>
### 3.2 - Single Step of Convolution 

In this part, implement a single step of convolution, in which you apply the filter to a single position of the input. This will be used to build a convolutional unit, which: 

- Takes an input volume 
- Applies a filter at every position of the input
- Outputs another volume (usually of different size)


在这一部分中，我们将实现单步卷积，即在输入的单个位置应用滤波器。这将用于建立一个卷积单元，它可以 

- 获取输入Volume 
- 在输入的每个位置应用滤波器
- 输出另一个volume（通常大小不同）
  
![Convolution_schematic](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/bb2c803b-9ee6-47fb-ae03-631939d29e91)

<caption><center> <u> <font color='purple'> <b>Figure 2</b> </u><font color='purple'>  : <b>Convolution operation</b><br> with a filter of 3x3 and a stride of 1 (stride = amount you move the window each time you slide) </center></caption>

In a computer vision application, each value in the matrix on the left corresponds to a single pixel value. You convolve a 3x3 filter with the image by multiplying its values element-wise with the original matrix, then summing them up and adding a bias. In this first step of the exercise, you will implement a single step of convolution, corresponding to applying a filter to just one of the positions to get a single real-valued output. 

Later in this notebook, you'll apply this function to multiple positions of the input to implement the full convolutional operation. 

在计算机视觉应用中，左边矩阵中的每个值都对应一个像素值。通过将 3x3 滤波器的值与原始矩阵逐元素相乘，然后求和并加上偏置，就可以将图像卷积为 3x3 滤波器。在练习的第一步，您将执行单步卷积，即在其中一个位置应用滤波器，以获得单个实值输出。

在本笔记本的后面部分，您将对输入的多个位置应用此函数，以实现完整的卷积操作。

<a name='ex-2'></a>
### Exercise 2 - conv_single_step
Implement `conv_single_step()`. 
    
[Hint](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sum.html).


**Note**: The variable b will be passed in as a numpy array.  If you add a scalar (a float or integer) to a numpy array, the result is a numpy array.  In the special case of a numpy array containing a single value, you can cast it as a float to convert it to a scalar.

**注意**： 变量 b 将作为一个 numpy 数组传入。 如果在一个 numpy 数组中加入一个标量（浮点数或整数），结果就是一个 numpy 数组。 在一个包含单个值的 numpy 数组的特殊情况下，你可以将其投为浮点数，将其转换为标量。

# GRADED FUNCTION: conv_single_step

**代码解释**
其中的a_slice_prev 的shape（f,f,n_C_prev） 就是在input那层进行取值的那个方块.
W 也是个array， shape是（f,f,n_C_prev）就是filter的weight parameters
b 是matrix of shape（1，1，1）

```python
def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    #(≈ 3 lines of code)
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    # s = None
    # Sum over all entries of the volume s.
    # Z = None
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    # Z = None
    # YOUR CODE STARTS HERE
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = Z + float(b)
    # YOUR CODE ENDS HERE

    return Z
```

**Test**
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/8d48be7c-a755-43d0-993e-b95d747fcf33)

<a name='3-3'></a>
### 3.3 - Convolutional Neural Networks - Forward Pass

In the forward pass, you will take many filters and convolve them on the input. Each 'convolution' gives you a 2D matrix output. You will then stack these outputs to get a 3D volume: 


https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/ed374291-34e7-4ece-968c-2889455c0a1a


### Exercise 3 -  conv_forward
Implement the function below to convolve the filters `W` on an input activation `A_prev`.  
This function takes the following inputs:
* `A_prev`, the activations output by the previous layer (for a batch of m inputs); 
* Weights are denoted by `W`.  The filter window size is `f` by `f`.
* The bias vector is `b`, where each filter has its own (single) bias. 

You also have access to the hyperparameters dictionary, which contains the stride and the padding. 

**Hint**: 
1. To select a 2x2 slice at the upper left corner of a matrix "a_prev" (shape (5,5,3)), you would do:
```python
a_slice_prev = a_prev[0:2,0:2,:]
```
Notice how this gives a 3D slice that has height 2, width 2, and depth 3.  Depth is the number of channels.  
This will be useful when you will define `a_slice_prev` below, using the `start/end` indexes you will define.

2. To define a_slice you will need to first define its corners `vert_start`, `vert_end`, `horiz_start` and `horiz_end`. This figure may be helpful for you to find out how each of the corners can be defined using h, w, f and s in the code below.


<caption><center> <u> <font color='purple'> <b>Figure 3</b> </u><font color='purple'>  : <b>Definition of a slice using vertical and horizontal start/end (with a 2x2 filter)</b> <br> This figure shows only a single channel.  </center></caption>
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/4b0be2cd-e989-4ba8-b84b-62f21e274d96)


**Reminder**:
    
The formulas relating the output shape of the convolution to the input shape are:
    
$$n_H = \Bigl\lfloor \frac{n_{H_{prev}} - f + 2 \times pad}{stride} \Bigr\rfloor +1$$
$$n_W = \Bigl\lfloor \frac{n_{W_{prev}} - f + 2 \times pad}{stride} \Bigr\rfloor +1$$
$$n_C = \text{number of filters used in the convolution}$$
    



For this exercise, don't worry about vectorization! Just implement everything with for-loops.


#### Additional Hints (if you're stuck):


* Use array slicing (e.g.`varname[0:1,:,3:5]`) for the following variables:  
  `a_prev_pad` ,`W`, `b`  
  - Copy the starter code of the function and run it outside of the defined function, in separate cells.  
  - Check that the subset of each array is the size and dimension that you're expecting.  
* To decide how to get the `vert_start`, `vert_end`, `horiz_start`, `horiz_end`, remember that these are indices of the previous layer.  
  - Draw an example of a previous padded layer (8 x 8, for instance), and the current (output layer) (2 x 2, for instance).  
  - The output layer's indices are denoted by `h` and `w`.  
* Make sure that `a_slice_prev` has a height, width and depth.
* Remember that `a_prev_pad` is a subset of `A_prev_pad`.  
  - Think about which one should be used within the for loops.


 这段代码定义了一个卷积神经网络（Convolutional Neural Network，简称 CNN）中的卷积前向传播函数。下面是对每一部分的解释：

conv_forward(A_prev, W, b, hparameters):: 这是函数定义，其中：

A_prev 是前一层的输出激活值，一个形状为 (m, n_H_prev, n_W_prev, n_C_prev) 的 numpy 数组。
W 是权重，一个形状为 (f, f, n_C_prev, n_C) 的 numpy 数组。
b 是偏差，一个形状为 (1, 1, 1, n_C) 的 numpy 数组。
hparameters 是一个包含 "stride" 和 "pad" 的 python 字典。
接下来的一部分代码从输入的数组 A_prev 和 W 中获取维度信息，并从 hparameters 中提取步长（stride）和填充（pad）信息。

然后，代码计算了卷积输出体积的维度，并初始化输出体积 Z 为零。

A_prev_pad 是通过对 A_prev 进行填充（padding）得到的。

随后的四个 for 循环是在批量训练样本、输出体积的垂直轴、水平轴和通道（即过滤器数量）上进行循环。在这个过程中，每个 "slice"（切片）的垂直和水平起始和结束位置被找出，从 a_prev_pad 中得到一个三维切片，然后与正确的滤波器 W 和偏差 b 进行卷积，得到一个输出神经元。

conv_single_step(a_slice_prev, weights, biases) 是一个未在这段代码中定义的函数，它执行的是单步卷积运算。

最后，将一些需要在后向传播中使用的信息保存在 "cache" 中，并返回卷积输出 Z 和 cache。

**代码逐行解释**

def conv_forward(A_prev, W, b, hparameters): 这一行定义了一个名为 conv_forward 的函数，它接受四个参数：前一层的输出 A_prev，权重 W，偏置 b，以及一个包含超参数（如步长和填充）的字典 hparameters。

(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape 这一行从输入 A_prev 中获取形状信息，并将各维度的大小分别赋值给 m, n_H_prev, n_W_prev 和 n_C_prev。

(f, f, n_C_prev, n_C) = W.shape 这一行从权重 W 中获取形状信息，并将各维度的大小分别赋值给 f, f, n_C_prev 和 n_C。

stride = hparameters['stride'] 和 pad = hparameters['pad'] 这两行从 hparameters 字典中提取步长和填充参数。

n_H = int((n_H_prev - f + 2 * pad) / stride) + 1 和 n_W = int((n_W_prev - f + 2 * pad) / stride) + 1 这两行根据公式计算卷积操作后输出的高度 n_H 和宽度 n_W。

Z = np.zeros((m, n_H, n_W, n_C)) 这一行创建一个全零的数组 Z，用于存储卷积结果。

A_prev_pad = np.pad(A_prev, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0)) 这一行对输入 A_prev 进行填充，创建一个新的数组 A_prev_pad。

接下来的部分包含了四层嵌套循环，对每个训练样本的每个输出通道的每个垂直和水平位置进行遍历：

for i in range(m): 这一行开始遍历所有的训练样本。
a_prev_pad = A_prev_pad[i] 这一行选取第 i 个训练样本的填充后的激活值。
for h in range(n_H): 这一行开始遍历输出体积的垂直轴。
vert_start = h * stride 和 vert_end = vert_start + f 这两行计算当前切片在垂直轴上的起始和结束位置。
for w in range(n_W): 这一行开始遍历输出体积的水平轴。
horiz_start = w * stride 和 horiz_end = horiz_start + f 这两行计算当前切片在水平轴上的起始和结束位置。
for c in range(n_C): 这一行开始遍历输出体积的通道（即过滤器数量）。
a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] 这一行从 a_prev_pad 中获取三维切片。
weights = W[...,c] 和 biases = b[...,c] 这两行获取当前通道的权重和偏置。
Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases) 这一行执行单步卷积运算，得到一个输出神经元，并将结果存入 Z。
最后的部分：

cache = (A_prev, W, b, hparameters) 这一行将一些在后向传播中需要用到的变量保存在 cache 中。
return Z, cache 这一行返回卷积的输出 Z 和 cache。

```python
# GRADED FUNCTION: conv_forward

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    # (m, n_H_prev, n_W_prev, n_C_prev) = None
    
    # Retrieve dimensions from W's shape (≈1 line)
    # (f, f, n_C_prev, n_C) = None
    
    # Retrieve information from "hparameters" (≈2 lines)
    # stride = None
    # pad = None
    
    # Compute the dimensions of the CONV output volume using the formula given above. 
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    # n_H = None
    # n_W = None
    
    # Initialize the output volume Z with zeros. (≈1 line)
    # Z = None
    
    # Create A_prev_pad by padding A_prev
    # A_prev_pad = None
    
    # for i in range(None):               # loop over the batch of training examples
        # a_prev_pad = None               # Select ith training example's padded activation
        # for h in range(None):           # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            # vert_start = None
            # vert_end = None
            
            # for w in range(None):       # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice" (≈2 lines)
                # horiz_start = None
                # horiz_end = None
                
                # for c in range(None):   # loop over channels (= #filters) of the output volume
                                        
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    # a_slice_prev = None
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                    # weights = None
                    # biases = None
                    # Z[i, h, w, c] = None
    # YOUR CODE STARTS HERE
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # Compute the dimensions of the CONV output volume using the formula given above. 
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
    # Initialize the output volume Z with zeros.
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = np.pad(A_prev, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0))

    for i in range(m):               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]               # Select ith training example's padded activation
        for h in range(n_H):           # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice"
            vert_start = h * stride
            vert_end = vert_start + f
            
            for w in range(n_W):       # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice"
                horiz_start = w * stride
                horiz_end = horiz_start + f
                
                for c in range(n_C):   # loop over channels (= #filters) of the output volume
                    # Use the corners to define the (3D) slice of a_prev_pad 
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    weights = W[...,c]
                    biases = b[...,c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
    
    # YOUR CODE ENDS HERE
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache
  ```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c9eaa527-f4e0-4487-be67-626db8bd19aa)


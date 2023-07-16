# 在不使用高级API keras的情况下，通过自定义函数构建神经网络.

Use tf.Variable to modify the state of a variable
Explain the difference between a variable and a constant
Train a Neural Network on a TensorFlow dataset
Programming frameworks like TensorFlow not only cut down on time spent coding, but can also perform optimizations that speed up the code itself.

## Table of Contents
- [1- Packages](#1)
    - [1.1 - Checking TensorFlow Version](#1-1)
- [2 - Basic Optimization with GradientTape](#2)
    - [2.1 - Linear Function](#2-1)
        - [Exercise 1 - linear_function](#ex-1)
    - [2.2 - Computing the Sigmoid](#2-2)
        - [Exercise 2 - sigmoid](#ex-2)
    - [2.3 - Using One Hot Encodings](#2-3)
        - [Exercise 3 - one_hot_matrix](#ex-3)
    - [2.4 - Initialize the Parameters](#2-4)
        - [Exercise 4 - initialize_parameters](#ex-4)
- [3 - Building Your First Neural Network in TensorFlow](#3)
    - [3.1 - Implement Forward Propagation](#3-1)
        - [Exercise 5 - forward_propagation](#ex-5)
    - [3.2 Compute the Total Loss](#3-2)
        - [Exercise 6 - compute_total_loss](#ex-6)
    - [3.3 - Train the Model](#3-3)
- [4 - Bibliography](#4)

<a name='1'></a>
## 1 - Packages

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time

<a name='1-1'></a>
### 1.1 - Checking TensorFlow Version 

You will be using v2.3 for this assignment, for maximum speed and efficiency.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/053ca842-502b-4262-a3af-e74158f2fe14)
<a name='2'></a>
## 2 - Basic Optimization with GradientTape

The beauty of TensorFlow 2 is in its simplicity. Basically, all you need to do is implement forward propagation through a computational graph. TensorFlow will compute the derivatives for you, by moving backwards through the graph recorded with `GradientTape`. All that's left for you to do then is specify the cost function and optimizer you want to use! 

When writing a TensorFlow program, the main object to get used and transformed is the `tf.Tensor`. These tensors are the TensorFlow equivalent of Numpy arrays, i.e. multidimensional arrays of a given data type that also contain information about the computational graph.

Below, you'll use `tf.Variable` to store the state of your variables. Variables can only be created once as its initial value defines the variable shape and type. Additionally, the `dtype` arg in `tf.Variable` can be set to allow data to be converted to that type. But if none is specified, either the datatype will be kept if the initial value is a Tensor, or `convert_to_tensor` will decide. It's generally best for you to specify directly, so nothing breaks!
<a name='2'></a>
## 2 - 梯度带的基本优化

TensorFlow 2的魅力在于其简单性。基本上，您需要做的就是通过计算图实现前向传播。TensorFlow将通过 "GradientTape "记录的图向后移动，为您计算导数。您所要做的就是指定您要使用的代价函数和优化器！

在编写TensorFlow程序时，使用和转换的主要对象是`tf.Tensor`。这些张量相当于TensorFlow的Numpy数组，即给定数据类型的多维数组，同时包含计算图的信息。

下面，您将使用`tf.Variable`来存储变量的状态。变量只能创建一次，因为它的初始值定义了变量的形状和类型。此外，可以设置`tf.Variable`中的`dtype`参数，以便将数据转换为该类型。但如果没有指定，如果初始值是张量，数据类型将被保留，或者由`convert_to_tensor`决定。一般来说，最好是直接指定，这样就不会出错！


Here you'll call the TensorFlow dataset created on a HDF5 file, which you can use in place of a Numpy array to store your datasets. You can think of this as a TensorFlow data generator! 

You will use the Hand sign data set, that is composed of images with shape 64x64x3.

这里你将调用在HDF5文件上创建的TensorFlow数据集，你可以用它来代替Numpy数组来存储你的数据集。您可以将其视为TensorFlow数据生成器！

您将使用手势数据集，该数据集由形状为64x64x3的图像组成。

## 加载数据，并把数据转换成Tensorflow需要的样子

这段代码使用了h5py库来读取HDF5文件格式中的训练数据和测试数据。

h5py.File('datasets/train_signs.h5', "r")：这行代码打开名为train_signs.h5的HDF5文件，并以只读模式打开它。HDF5是一种用于存储和组织大型数据集的文件格式，通常在机器学习中用于存储训练和测试数据。

h5py.File('datasets/test_signs.h5', "r")：这行代码打开名为test_signs.h5的HDF5文件，并以只读模式打开它，用于存储测试数据。

一旦这两个HDF5文件被打开，你就可以使用train_dataset和test_dataset两个变量来访问其中的数据集和相关信息。通常，这些文件会包含训练数据集和测试数据集的特征（例如图像数据）以及对应的标签（例如图像所属的类别）。

要进一步使用这些数据集，你可以通过h5py库中的方法来获取其中的数据和元数据。例如，可以使用类似train_dataset['features']的方式来获取训练数据集中的特征数据，train_dataset['labels']来获取训练数据集中的标签数据，等等。

train_dataset = h5py.File('datasets/train_signs.h5', "r")
test_dataset = h5py.File('datasets/test_signs.h5', "r")

**进行提取和转换**

这段代码使用TensorFlow中的tf.data.Dataset.from_tensor_slices方法从numpy数组（或张量）中创建了训练和测试数据集的tf.data.Dataset对象。

train_dataset['train_set_x']：这部分代码从train_dataset对象中获取了名为train_set_x的数据，这可能是训练数据集的特征数据（例如图像数据）。

train_dataset['train_set_y']：这部分代码从train_dataset对象中获取了名为train_set_y的数据，这可能是训练数据集的标签数据（例如图像类别标签）。

test_dataset['test_set_x']：这部分代码从test_dataset对象中获取了名为test_set_x的数据，这可能是测试数据集的特征数据。

test_dataset['test_set_y']：这部分代码从test_dataset对象中获取了名为test_set_y的数据，这可能是测试数据集的标签数据。

x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

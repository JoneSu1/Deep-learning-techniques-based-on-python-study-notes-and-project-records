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
```
train_dataset = h5py.File('datasets/train_signs.h5', "r")
test_dataset = h5py.File('datasets/test_signs.h5', "r")
```
**进行提取和转换**

这段代码使用TensorFlow中的tf.data.Dataset.from_tensor_slices方法从numpy数组（或张量）中创建了训练和测试数据集的tf.data.Dataset对象。

train_dataset['train_set_x']：这部分代码从train_dataset对象中获取了名为train_set_x的数据，这可能是训练数据集的特征数据（例如图像数据）。

train_dataset['train_set_y']：这部分代码从train_dataset对象中获取了名为train_set_y的数据，这可能是训练数据集的标签数据（例如图像类别标签）。

test_dataset['test_set_x']：这部分代码从test_dataset对象中获取了名为test_set_x的数据，这可能是测试数据集的特征数据。

test_dataset['test_set_y']：这部分代码从test_dataset对象中获取了名为test_set_y的数据，这可能是测试数据集的标签数据。
```python
x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])
```

Since TensorFlow Datasets are generators, you can't access directly the contents unless you iterate over them in a for loop, or by explicitly creating a Python iterator using `iter` and consuming its
elements using `next`. Also, you can inspect the `shape` and `dtype` of each element using the `element_spec` attribute.

由于TensorFlow数据集是生成器，你不能直接访问其内容，除非你在一个for循环中遍历它们，或者通过使用`iter`显式地创建一个Python迭代器，并使用`next`消耗它的
元素。另外，你可以使用`element_spec`属性检查每个元素的`shape`和`dtype`。
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/479da84a-4625-4a23-b241-30955be6ef2c)


The dataset that you'll be using during this assignment is a subset of the sign language digits. It contains six different classes representing the digits from 0 to 5.

本作业中您将使用的数据集是手语数字的子集。它包含六个不同的类别，代表从0到5的数字。


这段代码通过遍历TensorFlow Dataset中的y_train（假设y_train是一个TensorFlow Dataset对象）来获取训练数据集中的唯一标签，并将其存储在一个集合（unique_labels）中。

unique_labels = set()：这一行创建了一个空的集合，用于存储唯一的标签。

for element in y_train:：这个循环遍历了y_train中的每个元素。

unique_labels.add(element.numpy())：在循环中，代码使用add方法将每个元素（假设是一个Tensor）的numpy表示（即元素的实际值）添加到unique_labels集合中。这样，集合unique_labels就会包含训练数据集中的所有唯一标签值。

print(unique_labels)：最后，代码打印输出了unique_labels集合，其中包含训练数据集中的所有唯一标签值。

总之，这段代码的作用是获取训练数据集y_train中的所有唯一标签，并将其打印输出。
```python
unique_labels = set()
for element in y_train:
    unique_labels.add(element.numpy())
print(unique_labels)
```

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c9e673dd-75a3-48ba-8d8d-d4a8403a7a71)

You can see some of the images in the dataset by running the following cell.

您可以通过运行以下单元格查看数据集中的部分图像。

这段代码使用matplotlib库在一个5x5的图像网格中显示训练数据集中的图像和对应的标签。

images_iter = iter(x_train)：这行代码将训练数据集x_train转换为一个迭代器对象images_iter。迭代器可以用于逐个访问数据集中的元素。

labels_iter = iter(y_train)：这行代码将训练数据集y_train转换为一个迭代器对象labels_iter。同样，迭代器可以用于逐个访问数据集中的元素。

plt.figure(figsize=(10, 10))：这行代码创建一个大小为10x10的新图像窗口。

for i in range(25):：这个循环遍历25次，即在图像网格中显示25张图像。

ax = plt.subplot(5, 5, i + 1)：这行代码创建一个子图，将当前图像放在5x5的网格中的第i+1个位置。

plt.imshow(next(images_iter).numpy().astype("uint8"))：这行代码使用next(images_iter)获取训练数据集中的下一个图像，并使用imshow方法显示图像。.numpy()方法将TensorFlow张量转换为NumPy数组，.astype("uint8")将数据类型转换为无符号8位整数。

plt.title(next(labels_iter).numpy().astype("uint8"))：这行代码使用next(labels_iter)获取训练数据集中下一个图像对应的标签，并使用title方法将其显示为图像的标题。同样，.numpy()方法将TensorFlow张量转换为NumPy数组，.astype("uint8")将数据类型转换为无符号8位整数。

plt.axis("off")：这行代码关闭图像的坐标轴显示，以便更好地查看图像本身。

```python
images_iter = iter(x_train)
labels_iter = iter(y_train)
plt.figure(figsize=(10, 10))
for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(next(images_iter).numpy().astype("uint8"))
    plt.title(next(labels_iter).numpy().astype("uint8"))
    plt.axis("off")
```
![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/e6cb31aa-333e-49da-9915-161ef115448e)

TensorFlow数据集和Numpy数组之间还有一个额外的区别： 如果您需要转换数据集，您需要调用`map`方法，将作为参数传递的函数应用到每个元素上。

## Normalization the images 

和在Numpy中的常规操作相同，像素的最大值的255，我们让每一个dim_layer中的元素都除以255，归一化到[0,1]的范围中，然后将3维array，
转换成1维array（64*64*3）.

而在TensorFlow 的上面，我们首先需要把将图像像素值转换为浮点数类型。再将像素值除以255，将其归一化到[0, 1]范围内。

而代码和Numpy中时，有所不同.
这是Numpy中： image = image.astype(np.float32) / 255.0
这是Tf中：    image = tf.cast(image, tf.float32) / 255.0

然后将这个3维数组转成1维：64*64*3
使用reshape（）函数达到效果    image = tf.reshape(image, [-1,])
```python
def normalize(image):
    """
    Transform an image into a tensor of shape (64 * 64 * 3, )
    and normalize its components.
    
    Arguments
    image - Tensor.
    
    Returns: 
    result -- Transformed tensor 
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1,])
    return image
```
然后这段代码使用了TensorFlow中的map方法对训练数据集x_train和测试数据集x_test中的每个图像应用normalize函数进行归一化处理。

x_train和x_test是TensorFlow的tf.data.Dataset对象，表示训练数据集和测试数据集。

normalize是一个函数，它是将图像进行归一化的函数，将像素值除以255来将像素值缩放到[0, 1]的范围内。

new_train = x_train.map(normalize)：这行代码使用map方法，将normalize函数应用于x_train数据集中的每个图像。这样，训练数据集中的每个图像都会被归一化处理，并存储在new_train数据集中。

new_test = x_test.map(normalize)：这行代码使用map方法，将normalize函数应用于x_test数据集中的每个图像。这样，测试数据集中的每个图像也会被归一化处理，并存储在new_test数据集中。

最终，new_train和new_test数据集中的每个图像都被归一化处理，以便后续的图像处理和深度学习模型的训练。这是利用tf.data.Dataset.map()方法对数据集中的元素进行预处理的常见用法。
```
new_train = x_train.map(normalize)
new_test = x_test.map(normalize)
```
使用element_spec后缀来查询新复制的训练数组的内容
```
new_train.element_spec
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/759f930a-fb1c-413b-b177-7be48855ebde)
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/56a1e4fe-812c-40ff-9ca2-2cf99e14cbc7)

<a name='2-1'></a>
### 2.1 - Linear Function

Let's begin this programming exercise by computing the following equation: $Y = WX + b$, where $W$ and $X$ are random matrices and b is a random vector. 

让我们从计算下列方程开始这个编程练习：𝑌=𝑋+𝑏 ，其中 𝑋 和 𝑋 是随机矩阵，b 是随机向量。

<a name='ex-1'></a>
### Exercise 1 - linear_function

Compute $WX + b$ where $W, X$, and $b$ are drawn from a random normal distribution. W is of shape (4, 3), X is (3,1) and b is (4,1). As an example, this is how to define a constant X with the shape (3,1):
```python
X = tf.constant(np.random.randn(3,1), name = "X")

```
Note that the difference between `tf.constant` and `tf.Variable` is that you can modify the state of a `tf.Variable` but cannot change the state of a `tf.constant`.

You might find the following functions helpful: 
- tf.matmul(..., ...) to do a matrix multiplication
- tf.add(..., ...) to do an addition
- np.random.randn(...) to initialize randomly

 练习 1 - 线性函数
计算 𝑋+𝑏，其中𝑋, 𝑏和 𝑏从随机正态分布中抽取。W 的形状为 (4,3)，X 为 (3,1)，b 为 (4,1)。举例说明，如何定义形状为(3,1)的常数X：

X = tf.constant(np.random.randn(3,1), name = "X")
请注意，tf.constant和tf.Variable的区别在于，您可以修改tf.Variable的状态，但不能改变tf.constant的状态。

您可能会发现以下函数很有用：

tf.matmul(...,...)进行矩阵乘法运算
tf.add(...,...)进行加法运算
np.random.randn(...)用于随机初始化


这段代码实现了一个线性函数，根据给定的初始化规则创建了随机张量，并计算出线性函数的输出。

np.random.seed(1)：这行代码设置了随机种子，以确保随机数的生成与预期结果一致。

X = tf.constant(np.random.randn(3,1), name = "X")：这行代码创建了一个名为X的常量张量，形状为(3, 1)，值为随机生成的服从标准正态分布的数字。

W = tf.constant(np.random.randn(4,3), name = "W")：这行代码创建了一个名为W的常量张量，形状为(4, 3)，值为随机生成的服从标准正态分布的数字。

b = tf.constant(np.random.randn(4,1), name = "b")：这行代码创建了一个名为b的常量张量，形状为(4, 1)，值为随机生成的服从标准正态分布的数字。

Y = tf.matmul(W, X) + b：这行代码计算了线性函数的输出，通过矩阵乘法tf.matmul将W和X相乘，然后加上b得到结果Y。

最后，函数返回输出张量Y作为结果。

请注意，这段代码使用了TensorFlow库来创建和计算张量。在这个函数中，tf.constant用于创建常量张量，tf.matmul用于矩阵乘法操作。


# GRADED FUNCTION: linear_function
```python
def linear_function():
    """
    Implements a linear function: 
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- Y = WX + b 
    """

    np.random.seed(1)
    
    """
    Note, to ensure that the "random" numbers generated match the expected results,
    please create the variables in the order given in the starting code below.
    (Do not re-arrange the order).
    """
    # (approx. 4 lines)
    # X = ...
    # W = ...
    # b = ...
    # Y = ...
    # YOUR CODE STARTS HERE
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.matmul(W, X) + b
    # YOUR CODE ENDS HERE
    return Y
```
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/89557687-bfa1-4001-9ed0-fe26c5ffe665)

<a name='2-2'></a>
### 2.2 - Computing the Sigmoid 
Amazing! You just implemented a linear function. TensorFlow offers a variety of commonly used neural network functions like `tf.sigmoid` and `tf.softmax`.

For this exercise, compute the sigmoid of z. 

In this exercise, you will: Cast your tensor to type `float32` using `tf.cast`, then compute the sigmoid using `tf.keras.activations.sigmoid`. 

<a name='ex-2'></a>
### Exercise 2 - sigmoid

Implement the sigmoid function below. You should use the following: 

- `tf.cast("...", tf.float32)`
- `tf.keras.activations.sigmoid("...")`

<a name='2-2'></a>
### 2.2 - 计算Sigmoid函数 
太棒了！你刚刚实现了一个线性函数。TensorFlow提供了各种常用的神经网络函数，如`tf.sigmoid`和`tf.softmax`。

在本练习中，计算z的sigmoid。

在本练习中，您将 使用`tf.cast`将张量转换为`float32`类型，然后使用`tf.keras.activations.sigmoid`计算sigmoid。

<a name='ex-2'></a>
### 练习 2 - sigmoid

实现下面的sigmoid函数。你应该使用下面的方法： 

- `tf.cast("...",tf.float32)`。
- `tf.keras.activations.sigmoid("...")`。


# GRADED FUNCTION: sigmoid
```python

def sigmoid(z):
    
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    a -- (tf.float32) the sigmoid of z
    """
    # tf.keras.activations.sigmoid requires float16, float32, float64, complex64, or complex128.
    
    # (approx. 2 lines)
    # z = ...
    # a = ...
    # YOUR CODE STARTS HERE
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)

    
    # YOUR CODE ENDS HERE
    return a
```



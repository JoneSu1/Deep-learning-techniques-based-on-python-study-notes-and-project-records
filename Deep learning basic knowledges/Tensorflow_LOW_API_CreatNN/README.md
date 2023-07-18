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

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/9e902533-c801-492c-8110-5a3ebb481cde)

<a name='2-3'></a>
### 2.3 - Using One Hot Encodings

Many times in deep learning you will have a $Y$ vector with numbers ranging from $0$ to $C-1$, where $C$ is the number of classes. If $C$ is for example 4, then you might have the following y vector which you will need to convert like this:
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/10a1ed1e-85c3-47b7-871d-082dfd6291fb)

This is called "one hot" encoding, because in the converted representation, exactly one element of each column is "hot" (meaning set to 1). To do this conversion in numpy, you might have to write a few lines of code. In TensorFlow, you can use one line of code: 

- [tf.one_hot(labels, depth, axis=0)](https://www.tensorflow.org/api_docs/python/tf/one_hot)

`axis=0` indicates the new axis is created at dimension 0

<a name='ex-3'></a>
### Exercise 3 - one_hot_matrix

Implement the function below to take one label and the total number of classes $C$, and return the one hot encoding in a column wise matrix. Use `tf.one_hot()` to do this, and `tf.reshape()` to reshape your one hot tensor! 

- `tf.reshape(tensor, shape)`

2.3 - 使用单热编码
在深度学习中，您经常会有一个向量，其中包含的数字从到 ，其中是类的数量。例如，如果是4，那么您可能会有如下的Y向量，您需要像这样进行转换： 


这就是所谓的 "one hot "编码，因为在转换后的表示中，每一列中正好有一个元素是 "hot "的（意思是设为1）。要在numpy中进行这种转换，您可能需要编写几行代码。在TensorFlow中，您只需要写一行代码即可：

tf.one_hot(labels, depth, axis=0)
axis=0表示在维度0创建新轴


练习3 - one_hot_matrix
执行下面的函数，获取一个标签和类的总数，并以列为单位的矩阵返回one_hot编码。使用tf.one_hot()来实现这个功能，使用tf.reshape()来重塑你的one hot张量！



在神经网络中，One-Hot 编码是一种常用的向量表示方法，用于表示离散型的分类或标签信息。它将每个类别或标签映射为一个由 0 和 1 组成的向量，其中只有一个元素为 1，其他元素为 0。这个元素的位置表示对应的类别或标签。

例如，假设有一个分类问题，共有三个类别：猫、狗和鸟。使用 One-Hot 编码时，可以将它们表示为以下向量：

猫：[1, 0, 0]
狗：[0, 1, 0]
鸟：[0, 0, 1]


**代码解释**

先定义了一个名为 one_hot_matrix 的函数，它接受两个参数：label（分类标签）和 depth（类别的数量）。
```
    one_hot = tf.reshape(tf.one_hot(label, depth, axis=0), shape=[-1, ])
```
这一行代码的作用是计算输入标签 label 的独热编码。首先，我们使用 tf.one_hot() 函数将 label 编码为独热向量，其中 depth 指定了类别的数量。然后，我们使用 tf.reshape() 对结果进行形状调整，将其转换为单列矩阵。shape=[-1, ] 表示我们将结果调整为一个未知行数、单列的形状。


# GRADED FUNCTION: one_hot_matrix
```python
def one_hot_matrix(label, depth=6):
    """
    Computes the one hot encoding for a single label
    
    Arguments:
        label --  (int) Categorical labels
        depth --  (int) Number of different classes that label can take
    
    Returns:
         one_hot -- tf.Tensor A single-column matrix with the one hot encoding.
    """
    # (approx. 1 line)
    # one_hot = None(None(None, None, None), shape=[-1, ])
    # YOUR CODE STARTS HERE
    one_hot = tf.reshape(tf.one_hot(label, depth, axis = 0),shape =  [-1, ])
    
    # YOUR CODE ENDS HERE
    return one_hot
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/029bf760-f8a0-47d1-8bb2-8015b134a622)

**下一步就是使用.map函数使得array中的每一个元素都被one_hot_matrix函数处理**
```python
new_y_test = y_test.map(one_hot_matrix)
new_y_train = y_train.map(one_hot_matrix)
```
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/586c9cbc-ed2d-47ff-b7c9-c43e91e917d4)

<a name='2-4'></a>
### 2.4 - Initialize the Parameters 

Now you'll initialize a vector of numbers with the Glorot initializer. The function you'll be calling is `tf.keras.initializers.GlorotNormal`, which draws samples from a truncated normal distribution centered on 0, with `stddev = sqrt(2 / (fan_in + fan_out))`, where `fan_in` is the number of input units and `fan_out` is the number of output units, both in the weight tensor. 

To initialize with zeros or ones you could use `tf.zeros()` or `tf.ones()` instead. 

<a name='ex-4'></a>
### Exercise 4 - initialize_parameters

Implement the function below to take in a shape and to return an array of numbers using the GlorotNormal initializer. 

 - `tf.keras.initializers.GlorotNormal(seed=1)`
 - `tf.Variable(initializer(shape=())`


<a name='2-4'></a> ### 2.4 - 初始化参数
### 2.4 - 初始化参数 

现在，您将使用Glorot初始化器初始化一个数字向量。你要调用的函数是`tf.keras.initializers.GlorotNormal`，它从以0为中心的截断正态分布中抽取样本，其中`stddev = sqrt(2 / (fan_in + fan_out))`，
`fan_in`是输入单位的数量，`fan_out`是输出单位的数量，两者都在权重张量中。

要使用0或1初始化，可以使用`tf.zeros()`或`tf.nes()`代替。

<a name='ex-4'></a>.
### 练习 4 - 初始化参数

实现下面的函数，使用GlorotNormal初始化器接收一个形状并返回一个数组。

 - `tf.keras.initializers.GlorotNormal(seed=1)`。
 - `tf.Variable(initializer(shape=())`。

#### 什么是GlorotNormal

Glorot initializer，也称为Xavier初始化器，是一种常用的权重初始化方法，用于初始化神经网络中的参数（权重）。它由Xavier Glorot和Yoshua Bengio在2010年提出，并被广泛应用于深度学习中。

Glorot初始化器的目标是在网络的不同层之间保持输入和输出的方差相等。它考虑了每个神经元的输入和输出连接数量，以及非线性激活函数的特性。

对于具有n_in个输入和n_out个输出的层，Glorot初始化器使用以下方法初始化权重：

对于均匀分布的权重初始化（uniform distribution），权重在[-limit, limit]之间均匀采样，其中limit = sqrt(6 / (n_in + n_out))。
对于正态分布的权重初始化（normal distribution），权重从均值为0，标准差为sqrt(2 / (n_in + n_out))的正态分布中采样。
通过使用适当的方差来初始化权重，Glorot初始化器有助于避免梯度消失或梯度爆炸问题，从而更好地训练深度神经网络。这种初始化方法在许多常见的深度学习框架中都被默认使用或作为一种选择提供。


**这是对每一个已经初始化过的参数，进行了glorot处理**

# GRADED FUNCTION: initialize_parameters
```python
def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
                                
    initializer = tf.keras.initializers.GlorotNormal(seed=1)   
    #(approx. 6 lines of code)
    # W1 = ...
    # b1 = ...
    # W2 = ...
    # b2 = ...
    # W3 = ...
    # b3 = ...
    # Initialize W1
    W1 = tf.Variable(initializer(shape=(25, 12288)))

    # Initialize b1
    b1 = tf.Variable(initializer((25, 1)))

    # Initialize W2
    W2 = tf.Variable(initializer(shape=(12, 25)))

    # Initialize b2
    b2 = tf.Variable(initializer(shape = (12, 1)))

    # Initialize W3
    W3 = tf.Variable(initializer(shape=(6, 12)))

    # Initialize b3
    b3 = tf.Variable(initializer((6, 1)))
    # YOUR CODE ENDS HERE

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters
```
    
<a name='3'></a>
## 3 - Building Your First Neural Network in TensorFlow

In this part of the assignment you will build a neural network using TensorFlow. Remember that there are two parts to implementing a TensorFlow model:

- Implement forward propagation
- Retrieve the gradients and train the model

Let's get into it!

<a name='3-1'></a>
### 3.1 - Implement Forward Propagation 

One of TensorFlow's great strengths lies in the fact that you only need to implement the forward propagation function and it will keep track of the operations you did to calculate the back propagation automatically.  


<a name='ex-5'></a>
### Exercise 5 - forward_propagation

Implement the `forward_propagation` function.

**Note** Use only the TF API. 

- tf.math.add
- tf.linalg.matmul
- tf.keras.activations.relu

You will not apply "softmax" here. You'll see below, in `Exercise 6`, how the computation for it can be done internally by TensorFlow.



# GRADED FUNCTION: forward_propagation
```python
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    #(approx. 5 lines)                   # Numpy Equivalents:
    # Z1 = ...                           # Z1 = np.dot(W1, X) + b1
    # A1 = ...                           # A1 = relu(Z1)
    # Z2 = ...                           # Z2 = np.dot(W2, A1) + b2
    # A2 = ...                           # A2 = relu(Z2)
    # Z3 = ...                           # Z3 = np.dot(W3, A2) + b3
    # YOUR CODE STARTS HERE
    Z1 = tf.linalg.matmul(W1, X) + b1
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.linalg.matmul(W2, A1) + b2
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.linalg.matmul(W3, A2) + b3
    # YOUR CODE ENDS HERE
    
    return Z3
```

<a name='3-2'></a>
### 3.2 Compute the Total Loss

All you have to do now is define the loss function that you're going to use. For this case, since we have a classification problem with 6 labels, a categorical cross entropy will work!

You are used to compute the cost value which sums the losses over the whole batch (i.e. all mini-batches) of samples, then divide the sum by the total number of samples. Here, you will achieve this in two steps. 

In step 1, the `compute_total_loss` function will only take care of summing the losses from one mini-batch of samples. Then, as you train the model (in section 3.3) which will call this `compute_total_loss` function once per mini-batch, step 2 will be done by accumulating the sums from each of the mini-batches, and finishing it with the division by the total number of samples to get the final cost value.

Computing the "total loss" instead of "mean loss" in step 1 can make sure the final cost value to be consistent. For example, if the mini-batch size is 4 but there are just 5 samples in the whole batch, then the last mini-batch is going to have 1 sample only. Considering the 5 samples, losses to be [0, 1, 2, 3, 4] respectively, we know the final cost should be their average which is 2. Adopting the "total loss" approach will get us the same answer. However, the "mean loss" approach will first get us 1.5 and 4 for the two mini-batches, and then finally 2.75 after taking average of them, which is different from the desired result of 2. Therefore, the "total loss" approach is adopted here. 

现在要做的就是定义要使用的损失函数。在本例中，由于我们面临的是一个有6个标签的分类问题，因此可以使用分类交叉熵！

您需要计算代价值，代价值为整批样本（即所有小批量样本）的损失总和，然后用总和除以样本总数。这里分两步实现。

在第一步中，compute_total_loss 函数将只计算一个迷你批次样本的损失总和。然后，当您训练模型时（在第 3.3 节中），每个迷你批次将调用一次 compute_total_loss 函数，第 2 步将通过累加每个迷你批次的总和来完成，最后除以样本总数得到最终的成本值。

在步骤1中计算 "总损失 "而不是 "平均损失 "可以确保最终成本值保持一致。例如，如果小批量为 4 个，但整批样品只有 5 个，那么最后一个小批量只有 1 个样品。考虑到这5个样品的损失分别为[0, 1, 2, 3, 4]，我们知道最终成本应该是它们的平均值，即2。但是，如果采用 "平均损失 "法，则两个小批量的成本分别为 1.5 和 4，取其平均值后，最终成本为 2.75，这与预期结果 2 不同。因此，这里采用 "总损失 "法。

<a name='ex-6'></a>
### Exercise 6 -  compute_total_loss

Implement the total loss function below. You will use it to compute the total loss of a batch of samples. With this convenient function, you can sum the losses across many batches, and divide the sum by the total number of samples to get the cost value. 
- It's important to note that the "`y_pred`" and "`y_true`" inputs of [tf.keras.losses.categorical_crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_crossentropy) are expected to be of shape (number of examples, num_classes). 

- `tf.reduce_sum` does the summation over the examples.

- You skipped applying "softmax" in `Exercise 5` which will now be taken care by the `tf.keras.losses.categorical_crossentropy` by setting its parameter `from_logits=True` (You can read the response by one of our mentors [here](https://community.deeplearning.ai/t/week-3-assignment-compute-total-loss-try-to-set-from-logits-false/243049/2?u=paulinpaloalto) in the Community for the mathematical reasoning behind it. If you are not part of the Community already, you can do so by going [here](https://www.coursera.org/learn/deep-neural-network/ungradedLti/ZE1VR/important-have-questions-issues-or-ideas-join-our-community).)
  

练习 6 - 计算总损失
实现下面的总损失函数。您将用它来计算一批样本的总损失。有了这个方便的函数，您就可以将许多批次的损失相加，然后将总和除以样本总数，得到成本值。

值得注意的是，tf.keras.lossings.categorical_crossentropy的输入 "y_pred "和 "y_true "预计为形状（示例数、类数）。

tf.reduce_sum将对示例求和。

您在练习 5 中跳过了应用 "softmax"，现在将由 tf.keras.losses.categorical_crossentropy 通过设置其参数 from_logits=True（您可以阅读我们的一位导师在社区中的回复，了解其背后的数学推理。如果您还不是社区的一员，您可以点击此处加入)。




# GRADED FUNCTION: compute_total_loss 
```python
def compute_total_loss(logits, labels):
    """
    Computes the total loss
    
    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
    labels -- "true" labels vector, same shape as Z3
    
    Returns:
    total_loss - Tensor of the total loss value
    """
    
    #(1 line of code)
    # remember to set `from_logits=True`
    # total_loss = ...
    # YOUR CODE STARTS HERE
    total_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True))
    
    # YOUR CODE ENDS HERE
    return total_loss

```

<a name='3-3'></a>
### 3.3 - Train the Model

Let's talk optimizers. You'll specify the type of optimizer in one line, in this case `tf.keras.optimizers.Adam` (though you can use others such as SGD), and then call it within the training loop. 

Notice the `tape.gradient` function: this allows you to retrieve the operations recorded for automatic differentiation inside the `GradientTape` block. Then, calling the optimizer method `apply_gradients`, will apply the optimizer's update rules to each trainable parameter. At the end of this assignment, you'll find some documentation that explains this more in detail, but for now, a simple explanation will do. ;) 


Here you should take note of an important extra step that's been added to the batch training process: 

- `tf.Data.dataset = dataset.prefetch(8)` 

What this does is prevent a memory bottleneck that can occur when reading from disk. `prefetch()` sets aside some data and keeps it ready for when it's needed. It does this by creating a source dataset from your input data, applying a transformation to preprocess the data, then iterating over the dataset the specified number of elements at a time. This works because the iteration is streaming, so the data doesn't need to fit into the memory. 

3.2 计算总损失
现在要做的就是定义要使用的损失函数。在本例中，由于我们面临的是一个有6个标签的分类问题，因此可以使用分类交叉熵！

您需要计算代价值，代价值为整批样本（即所有小批量样本）的损失总和，然后用总和除以样本总数。这里分两步实现。

在第一步中，compute_total_loss 函数将只计算一个迷你批次样本的损失总和。然后，当您训练模型时（在第 3.3 节中），每个迷你批次将调用一次 compute_total_loss 函数，第 2 步将通过累加每个迷你批次的总和来完成，最后除以样本总数得到最终的成本值。

在步骤1中计算 "总损失 "而不是 "平均损失 "可以确保最终成本值保持一致。例如，如果小批量为 4 个，但整批样品只有 5 个，那么最后一个小批量只有 1 个样品。考虑到这5个样品的损失分别为[0, 1, 2, 3, 4]，我们知道最终成本应该是它们的平均值，即2。然而，采用 "平均损失 "的方法将首先得到两个小批量的 1.5 和 4，然后在求出它们的平均值后得到 2.75，这与期望的结果 2 不同。因此，这里采用 "总损失 "的方法。


练习 6 - 计算总损失
实现下面的总损失函数。您将用它来计算一批样本的总损失。有了这个方便的函数，您就可以将许多批次的损失相加，然后将总和除以样本总数，得到成本值。

值得注意的是，tf.keras.lossings.categorical_crossentropy的输入 "y_pred "和 "y_true "预计为形状（示例数、类数）。

tf.reduce_sum将对示例求和。

您在练习 5 中跳过了应用 "softmax"，现在将由 tf.keras.losses.categorical_crossentropy 通过设置其参数 from_logits=True（您可以阅读我们的一位导师在社区中的回复，了解其背后的数学推理。如果您还不是社区的一员，您可以点击此处加入)。


**Disputed results**

# GRADED FUNCTION: compute_total_loss 
```python
def compute_total_loss(logits, labels):
    """
    Computes the total loss
    
    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
    labels -- "true" labels vector, same shape as Z3
    
    Returns:
    total_loss - Tensor of the total loss value
    """
    
    #(1 line of code)
    # remember to set `from_logits=True`
    # total_loss = ...
    # YOUR CODE STARTS HERE
    total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True))

    # YOUR CODE ENDS HERE
    return total_loss
```



<a name='3-3'></a>
### 3.3 - Train the Model

Let's talk optimizers. You'll specify the type of optimizer in one line, in this case `tf.keras.optimizers.Adam` (though you can use others such as SGD), and then call it within the training loop. 

Notice the `tape.gradient` function: this allows you to retrieve the operations recorded for automatic differentiation inside the `GradientTape` block. Then, calling the optimizer method `apply_gradients`, will apply the optimizer's update rules to each trainable parameter. At the end of this assignment, you'll find some documentation that explains this more in detail, but for now, a simple explanation will do. ;) 


Here you should take note of an important extra step that's been added to the batch training process: 

- `tf.Data.dataset = dataset.prefetch(8)` 

What this does is prevent a memory bottleneck that can occur when reading from disk. `prefetch()` sets aside some data and keeps it ready for when it's needed. It does this by creating a source dataset from your input data, applying a transformation to preprocess the data, then iterating over the dataset the specified number of elements at a time. This works because the iteration is streaming, so the data doesn't need to fit into the memory. 

```python
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    costs = []                                        # To keep track of the cost
    train_acc = []
    test_acc = []
    
    # Initialize your parameters
    #(1 line)
    parameters = initialize_parameters()

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # The CategoricalAccuracy will track the accuracy for this multiclass problem
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))
    
    # We can get the number of elements of a dataset using the cardinality method
    m = dataset.cardinality().numpy()
    
    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)
    #X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)# <<< extra step    
    #Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) # loads memory faster 

    # Do the training loop
    for epoch in range(num_epochs):

        epoch_total_loss = 0.
        
        #We need to reset object to start measuring from 0 the accuracy each epoch
        train_accuracy.reset_states()
        
        for (minibatch_X, minibatch_Y) in minibatches:
            
            with tf.GradientTape() as tape:
                # 1. predict
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)

                # 2. loss
                minibatch_total_loss = compute_total_loss(Z3, tf.transpose(minibatch_Y))

            # We accumulate the accuracy of all the batches
            train_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
            
            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_total_loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_total_loss += minibatch_total_loss
        
        # We divide the epoch total loss over the number of samples
        epoch_total_loss /= m

        # Print the cost every 10 epochs
        if print_cost == True and epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_total_loss))
            print("Train accuracy:", train_accuracy.result())
            
            # We evaluate the test set every 10 epochs to avoid computational overhead
            for (minibatch_X, minibatch_Y) in test_minibatches:
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)
                test_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
            print("Test_accuracy:", test_accuracy.result())

            costs.append(epoch_total_loss)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_states()


    return parameters, costs, train_acc, test_acc
```

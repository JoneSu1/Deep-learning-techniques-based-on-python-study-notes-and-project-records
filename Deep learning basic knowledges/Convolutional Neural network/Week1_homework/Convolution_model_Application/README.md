# Convolutional Neural Networks: Application

Welcome to Course 4's second assignment! In this notebook, you will:

- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API

**After this assignment you will be able to:**

- Build and train a ConvNet in TensorFlow for a __binary__ classification problem
- Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
- Explain different use cases for the Sequential and Functional APIs

To complete this assignment, you should already be familiar with TensorFlow. If you are not,
please refer back to the **TensorFlow Tutorial** of the third week of Course 2 
("**Improving deep neural networks**").

## Table of Contents

- [1 - Packages](#1)
    - [1.1 - Load the Data and Split the Data into Train/Test Sets](#1-1)
- [2 - Layers in TF Keras](#2)
- [3 - The Sequential API](#3)
    - [3.1 - Create the Sequential Model](#3-1)
        - [Exercise 1 - happyModel](#ex-1)
    - [3.2 - Train and Evaluate the Model](#3-2)
- [4 - The Functional API](#4)
    - [4.1 - Load the SIGNS Dataset](#4-1)
    - [4.2 - Split the Data into Train/Test Sets](#4-2)
    - [4.3 - Forward Propagation](#4-3)
        - [Exercise 2 - convolutional_model](#ex-2)
    - [4.4 - Train the Model](#4-4)
- [5 - History Object](#5)
- [6 - Bibliography](#6)


<a name='1'></a>
## 1 - Packages

As usual, begin by loading in the packages.
```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

%matplotlib inline
np.random.seed(1)
```

<a name='1-1'></a>
### 1.1 - Load the Data and Split the Data into Train/Test Sets

You'll be using the Happy House dataset for this part of the assignment, which contains images of peoples' faces. Your task will be to build a ConvNet that determines whether the people in the images are smiling or not -- because they only get to enter the house if they're smiling!  

```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

You can display the images contained in the dataset. Images are **64x64** pixels in RGB format (3 channels).
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/f15a4442-69e3-4a7e-a4a4-419c10d437f1)

```python
index = 124
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()
```
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/db630ad0-dee8-4108-8e36-175a81c5f638)

<a name='2'></a>
## 2 - Layers in TF Keras 

In the previous assignment, you created layers manually in numpy. In TF Keras, you don't have to write code directly to create layers. Rather, TF Keras has pre-defined layers you can use. 

When you create a layer in TF Keras, you are creating a function that takes some input and transforms it into an output you can reuse later. Nice and easy! 

<a name='3'></a>
## 3 - The Sequential API

In the previous assignment, you built helper functions using `numpy` to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call. Keras is a high-level abstraction built on top of TensorFlow, which allows for even more simplified and optimized model creation and training. 

For the first part of this assignment, you'll create a model using TF Keras' Sequential API, which allows you to build layer by layer, and is ideal for building models where each layer has **exactly one** input tensor and **one** output tensor. 

As you'll see, using the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks. Later in this notebook you'll spend some time building with a more flexible, powerful alternative: the Functional API. 


<a name='3-1'></a>
### 3.1 - Create the Sequential Model

As mentioned earlier, the TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order. 

You can also add layers incrementally to a Sequential model with the `.add()` method, or remove them using the `.pop()` method, much like you would in a regular Python list.

Actually, you can think of a Sequential model as behaving like a list of layers. Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.  If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn't be the right choice!

For any layer construction in Keras, you'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. Sequential models can be created by passing a list of layers to the Sequential constructor, like you will do in the next assignment.

<a name='ex-1'></a>
### Exercise 1 - happyModel

Implement the `happyModel` function below to build the following model: `ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Take help from [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 

Also, plug in the following parameters for all the steps:

 - [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): padding 3, input shape 64 x 64 x 3
 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 32 7x7 filters, stride 1
 - [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): for axis 3
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Using default parameters
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 1 neuron and a sigmoid activation. 
 
 
 **Hint:**
 
 Use **tfl** as shorthand for **tensorflow.keras.layers**
 

```python
def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
            # This ZeroPadding2D layer adds zeros at the boundaries of the input image.
            # Padding is set to 3, meaning 3 rows/columns of zeros are added to each side of the image.
            # The input shape is specified as 64x64 pixels with 3 color channels.
            tf.keras.layers.ZeroPadding2D(padding=3, input_shape=(64, 64, 3)),

            # This Conv2D layer performs a 2D convolution operation on the input.
            # It uses 32 filters of size 7x7, and moves the filters by 1 pixel step (stride) at a time.
            tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1)),
            
            # The BatchNormalization layer normalizes the activations of the previous layer 
            # at each batch (i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1).
            # We specify axis=3 to normalize the color channels of the images.
            tf.keras.layers.BatchNormalization(axis=3),
            
            # The ReLU layer applies the ReLU activation function to the output of the previous layer.
            # This activation function replaces all negative pixel values in the previous layer's output with zero.
            tf.keras.layers.ReLU(),
            
            # The MaxPooling2D layer performs max pooling operation, which reduces the dimensionality of the images.
            # It does this by only retaining the maximum value for each filtered area, and removing the remaining values.
            tf.keras.layers.MaxPooling2D(),
            
            # The Flatten layer flattens the 2D arrays into 1D arrays to prepare the data for the following Dense layer.
            tf.keras.layers.Flatten(),
            
            # The Dense layer is a neural network layer that is connected deeply, which means each neuron in the Dense layer 
            # is connected to all neurons in the previous layer. It has 1 unit for output with 'sigmoid' activation.
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
    
    return model
```
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/f725949c-eba1-498a-ba8b-a2a65ebd20ff)

Now that your model is created, you can compile it for training with an optimizer and loss of your choice. When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier! If you'd like to read more on how the compiler operates, check the docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).


模型创建完成后，您可以使用优化器和损失函数对其进行编译训练。当字符串准确度被指定为指标时，使用的准确度类型将根据所使用的损失函数自动转换。这是 TensorFlow 内置的众多优化之一，能让你的生活更轻松！如果您想进一步了解编译器的工作原理，请点击此处查看文档。

**构建 compile（编辑器）**
```python
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

It's time to check your model's parameters with the `.summary()` method. This will display the types of layers you have, the shape of the outputs, and how many parameters are in each layer. 

![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a7fbe22c-194b-41fc-b556-d1b02337e54a)


<a name='3-2'></a>
### 3.2 - Train and Evaluate the Model

After creating the model, compiling it with your choice of optimizer and loss function, and doing a sanity check on its contents, you are now ready to build! 

Simply call `.fit()` to train. That's it! No need for mini-batching, saving, or complex backpropagation computations. That's all been done for you, as you're using a TensorFlow dataset with the batches specified already. You do have the option to specify epoch number or minibatch size if you like (for example, in the case of an un-batched dataset).

<a name='3-2'></a>
#### 3.2 - 训练和评估模型

创建模型、使用您选择的优化器和损失函数编译模型并对其内容进行正确性检查后，您现在就可以构建模型了！

只需调用 `.fit()` 进行训练即可。就是这样！无需迷你批处理、保存或复杂的反向传播计算。这一切都已为您完成，因为您使用的是已指定批次的 TensorFlow 数据集。如果你愿意，还可以选择指定历元数或迷你批次大小（例如，在无批次数据集的情况下）。

**这里面的batch_size是指在进行mini——batch处理时候进行分割的**

batch_size 为16，那么数据将被分成许多大小为16的小批次（mini-batches）。也就是说，我们将会每次取16个样本来训练网络，并根据这16个样本的平均损失更新模型的权重和偏置。

```python
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c6018476-cf1e-4831-9e14-4335afc44fcd)

After that completes, just use `.evaluate()` to evaluate against your test set. This function will print the value of the loss function and the performance metrics specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.

完成后，只需使用 `.evaluate()` 根据测试集进行评估。该函数将打印损失函数的值和编译模型时指定的性能指标。在本例中，分别是 "二元交叉熵 "和 "准确度"。
```python
happy_model.evaluate(X_test, Y_test)
```
![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/77b4db6b-6d19-41f4-a3fe-f9d4936e784e)

Easy, right? But what if you need to build a model with shared layers, branches, or multiple inputs and outputs? This is where Sequential, with its beautifully simple yet limited functionality, won't be able to help you. 

Next up: Enter the Functional API, your slightly more complex, highly flexible friend.  

<a name='4'></a>
## 4 - The Functional API

Welcome to the second half of the assignment, where you'll use Keras' flexible [Functional API](https://www.tensorflow.org/guide/keras/functional) to build a ConvNet that can differentiate between 6 sign language digits. 

The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs. Imagine that, where the Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows much more flexibility. Where Sequential is a straight line, a Functional model is a graph, where the nodes of the layers can connect in many more ways than one. 

In the visual example below, the one possible direction of the movement Sequential model is shown in contrast to a skip connection, which is just one of the many ways a Functional model can be constructed. A skip connection, as you might have guessed, skips some layer in the network and feeds the output to a later layer in the network. Don't worry, you'll be spending more time with skip connections very soon! 


欢迎来到作业的后半部分，您将使用 Keras 灵活的 [Functional API](https://www.tensorflow.org/guide/keras/functional)，构建一个能够区分 6 位手语数字的 ConvNet。

功能 API 可以处理具有非线性拓扑结构、共享层以及具有多个输入或输出的层的模型。试想一下，序列应用程序接口要求模型以线性方式在各层中移动，而功能应用程序接口则允许更大的灵活性。顺序模型是一条直线，而功能模型则是一张图，图中各层的节点可以多种方式连接。

在下面的可视化示例中，显示的是顺序模型的一种可能运动方向，而跳过连接只是构建功能模型的多种方式之一。正如你可能已经猜到的那样，跳转连接跳过了网络中的某一层，并将输出馈送到网络的后一层。别担心，你很快就会花更多时间来学习跳转连接！

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/5fd2e990-8330-454e-b30a-ce96b5df8a02)

<a name='4-1'></a>
### 4.1 - Load the SIGNS Dataset

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.

<a name='4-1'></a>
#### 4.1 - 加载 SIGNS 数据集

请注意，SIGNS 数据集是由 6 个符号组成的集合，分别代表 0 到 5 的数字。
```python
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/5d7ef9ce-3642-4e6d-9ba1-bceb78eeeeb2)

下一个单元格将向您展示数据集中带标签图像的示例。请随意更改下面的索引值并重新运行，以查看不同的示例。
```python
# Example of an image from the dataset
index = 9
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
```
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/cd84ea2f-b5cb-4c51-8ded-d07d60526b4d)


<a name='4-2'></a>
### 4.2 - Split the Data into Train/Test Sets

In Course 2, you built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.

To get started, let's examine the shapes of your data. 

<a name='4-2'></a>
#### 4.2 - 将数据拆分为训练/测试集

在课程 2 中，您为该数据集构建了一个全连接网络。但由于这是一个图像数据集，因此更自然的做法是将 ConvNet 应用于该数据集。

首先，让我们检查一下数据的形状。
```python
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/123261c2-03db-46ff-a3d1-bce6a30c53c1)

<a name='4-3'></a>
### 4.3 - Forward Propagation

In TensorFlow, there are built-in functions that implement the convolution steps for you. By now, you should be familiar with how TensorFlow builds computational graphs. In the [Functional API](https://www.tensorflow.org/guide/keras/functional), you create a graph of layers. This is what allows such great flexibility.

However, the following model could also be defined using the Sequential API since the information flow is on a single line. But don't deviate. What we want you to learn is to use the functional API.

Begin building your graph of layers by creating an input node that functions as a callable object:
- **input_img = tf.keras.Input(shape=input_shape):** 

Then, create a new node in the graph of layers by calling a layer on the `input_img` object: 

- **tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img):** Read the full documentation on [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).

- **tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'):** `MaxPool2D()` downsamples your input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window.  For max pooling, you usually operate on a single example at a time and a single channel at a time. Read the full documentation on [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).

- **tf.keras.layers.ReLU():** computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU).

- **tf.keras.layers.Flatten()**: given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.  

    * If a tensor P has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), where $k=h \times w \times c$.  "k" equals the product of all the dimension sizes other than the first dimension.
    
    * For example, given a tensor with dimensions [100, 2, 3, 4], it flattens the tensor to be of shape [100, 24], where 24 = 2 * 3 * 4.  You can read the full documentation on [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten).

- **tf.keras.layers.Dense(units= ... , activation='softmax')(F):** given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

In the last function above (`tf.keras.layers.Dense()`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Lastly, before creating the model, you'll need to define the output using the last of the function's compositions (in this example, a Dense layer): 

- **outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)**


#### Window, kernel, filter, pool

The words "kernel" and "filter" are used to refer to the same thing. The word "filter" accounts for the amount of "kernels" that will be used in a single convolution layer. "Pool" is the name of the operation that takes the max or average value of the kernels. 

This is why the parameter `pool_size` refers to `kernel_size`, and you use `(f,f)` to refer to the filter size. 

Pool size and kernel size refer to the same thing in different objects - They refer to the shape of the window where the operation takes place. 


<a name='ex-2'></a>
### Exercise 2 - convolutional_model

Implement the `convolutional_model` function below to build the following model: `CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Use the functions above! 

Also, plug in the following parameters for all the steps:

 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 8 4 by 4 filters, stride 1, padding is "SAME"
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - **Conv2D**: Use 16 2 by 2 filters, stride 1, padding is "SAME"
 - **ReLU**
 - **MaxPool2D**: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 6 neurons and a softmax activation.

```python
def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_shape -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tf.keras.layers.Conv2D(8, (4, 4), strides=(1, 1), padding='same')(input_img)
    ## RELU
    A1 = tf.keras.layers.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.keras.layers.MaxPooling2D((8, 8), strides=(8, 8), padding='same')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tf.keras.layers.Conv2D(16, (2, 2), strides=(1, 1), padding='same')(P1)
    ## RELU
    A2 = tf.keras.layers.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.keras.layers.MaxPooling2D((4, 4), strides=(4, 4), padding='same')(A2)
    ## FLATTEN
    F = tf.keras.layers.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tf.keras.layers.Dense(6, activation='softmax')(F)
    
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
```
```python
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)
```
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/ca3ba18f-94f5-436b-9ff5-11e7eb0f86cf)


Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model! 

顺序型和功能型 API 都会返回一个 TF Keras 模型对象。唯一的区别在于对象模型内部如何处理输入！

<a name='4-4'></a>
### 4.4 - Train the Model


这里使用Keras 进行训练喝使用Sequential进行训练最大的区别就是在Sequential中可以直接使用input的数据，并且不用单独进行分割batch。

创建训练和测试数据集：tf.data.Dataset.from_tensor_slices 是 TensorFlow 的一个函数，用于创建一个数据集对象。这个函数接收一个元组作为输入，元组中的第一个元素是特征数据（在这个例子中是 X_train 和 X_test），第二个元素是标签数据（在这个例子中是 Y_train 和 Y_test）。数据集对象表示的是一个元素序列，在这里每个元素是一个样本和它对应的标签。

分批数据：batch(64) 函数将数据集分成多个批次，每个批次包含64个元素。在训练神经网络时，我们通常不会一次性处理所有的数据，而是分批次处理，每次处理一个批次的数据。

拟合模型：fit 函数用来训练模型。它需要接收输入数据和标签数据，以及训练的周期数（epochs）。在这个例子中，模型用训练数据集 train_dataset 训练了100个周期。另外，validation_data 参数设置了用于在训练过程中进行模型性能评估的数据集（在这个例子中是 test_dataset）。

fit 函数返回一个 history 对象，它包含了训练过程中的一些信息，例如每个周期的训练和验证损失等。

```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
```
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c164e0ac-5151-4eac-a3d7-d08d681283c9)

<a name='5'></a>
## 5 - History Object 

The history object is an output of the `.fit()` operation, and provides a record of all the loss and metric values in memory. It's stored as a dictionary that you can retrieve at `history.history`: 

5 - 历史对象
历史对象是 .fit() 操作的输出，它提供了内存中所有损耗和度量值的记录。它以字典形式存储，可在 history.history.History 中检索：
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c3810e90-613a-4365-bb11-fd6e112f7d55)


Now visualize the loss over time using `history.history`: 
```python
# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on. 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```

**代码解释**
提取历史记录: history.history 包含了训练过程中每个周期的训练损失、训练准确率、验证损失和验证准确率。它是一个字典，键是度量名称（如"loss", "accuracy", "val_loss", "val_accuracy"），值是对应的度量值列表。

创建数据框: pd.DataFrame(history.history) 将 history.history 字典转换为 Pandas 数据框。这样做的目的是方便后续的数据处理和可视化。

分别提取损失和准确率: df_loss = df_loss_acc[['loss','val_loss']] 和 df_acc = df_loss_acc[['accuracy','val_accuracy']] 分别提取出训练和验证的损失和准确率。

重命名列名: df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True) 和 df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True) 将列名重命名，使其更具可读性。

绘制图形: df_loss.plot() 和 df_acc.plot() 分别绘制了损失和准确率的图形。标题、图形尺寸、x轴和y轴标签都被设置了。
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/809339d8-40ef-486d-8dc5-5418e8e60992)

![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/bfd59b11-92bb-4023-9284-7405b2ae1349)


Congratulations! You've finished the assignment and built two models: One that recognizes smiles, and another that recognizes SIGN language with almost 80% accuracy on the test set. In addition to that, you now also understand the applications of two Keras APIs: Sequential and Functional. Nicely done!

By now, you know a bit about how the Functional API works and may have glimpsed the possibilities. In your next assignment, you'll really get a feel for its power when you get the opportunity to build a very deep ConvNet, using ResNets!

<a name='6'></a>
## 6 - Bibliography

You're always encouraged to read the official documentation. To that end, you can find the docs for the Sequential and Functional APIs here: 

https://www.tensorflow.org/guide/keras/sequential_model

https://www.tensorflow.org/guide/keras/functional

# Regularization
Deep Learning models have so much flexibility and capacity that **overfitting can be a serious problem**,
if the training dataset is not big enough. Sure it does well on the training set, but the learned network **doesn't generalize to new examples** that it has never seen!

**主要专注于L2 Regularization 和 Dropout**

**利用守门员踢球位置预测来应用L2和Dropout Regularization**
<a name='1'></a>
## 1 - Packages
```python
# import packages
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
from testCases import *
from public_tests import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2
```
<a name='2'></a>
## 2 - Problem Statement
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/36982bcb-bf56-4e80-81f3-3bb259c31008)

<a name='3'></a>
## 3 - Loading the Dataset
```
train_X, train_Y, test_X, test_Y = load_2D_dataset()
```
Each dot corresponds to a position on the football field where a football player has hit the ball with his/her head after the French goal keeper has shot the ball from the left side of the football field.
- If the dot is blue, it means the French player managed to hit the ball with his/her head
- If the dot is red, it means the other team's player hit the ball with their head

**Your goal**: Use a deep learning model to find the positions on the field where the goalkeeper should kick the ball.

每个点对应足球场上法国守门员从足球场左侧射门后足球运动员用头击球的位置。
- 如果点是蓝色的，表示法国队球员成功地用头顶到了球。
- 如果点是红色的，说明对方球员用头碰到了球。

**您的目标** 使用深度学习模型找出守门员应该踢球的位置。

**Analysis of the dataset**: This dataset is a little noisy, but it looks like a diagonal line separating the upper left half (blue) from the lower right half (red) would work well. 

You will first try a non-regularized model. Then you'll learn how to regularize it and decide which model you will choose to solve the French Football Corporation's problem. 

**数据集分析**： 这个数据集有点嘈杂，但看起来用一条对角线将左上半部分（蓝色）和右下半部分（红色）分开效果不错。

您将首先尝试非正则化模型。然后您将学习如何对其进行正则化，并决定选择哪种模型来解决法国足球公司的问题。

<a name='4'></a>
## 4 - Non-Regularized Model

You will use the following neural network (already implemented for you below). This model can be used:
- in *regularization mode* -- by setting the `lambd` input to a non-zero value. We use "`lambd`" instead of "`lambda`" because "`lambda`" is a reserved keyword in Python. 
- in *dropout mode* -- by setting the `keep_prob` to a value less than one

You will first try the model without any regularization. Then, you will implement:
- *L2 regularization* -- functions: "`compute_cost_with_regularization()`" and "`backward_propagation_with_regularization()`"
- *Dropout* -- functions: "`forward_propagation_with_dropout()`" and "`backward_propagation_with_dropout()`"

In each part, you will run this model with the correct inputs so that it calls the functions you've implemented. Take a look at the code below to familiarize yourself with the model.


<a name='4'></a>
## 4 - 非规则化模型

您将使用下面的神经网络（已经在下面为您实现）。这个模型可以
- 在*正则化模式*下--通过将`lambd`输入设置为非零值。我们使用"`lambd`"而不是"`lambda`"，因为"`lambda`"在Python中是一个保留关键字。
- 在*dropout模式*下--通过设置`keep_prob`为小于1的值

您将首先在没有任何正则化的情况下尝试模型。然后，您将实现
- *L2 正则化* -- 函数： "compute_cost_with_regularization()`"和 "backward_propagation_with_regularization()`"。
- *Dropout* -- 函数： "`forward_propagation_with_dropout()`" 和 "`backward_propagation_with_dropout()`" 函数

在每一部分中，您将以正确的输入运行该模型，以便它调用您实现的函数。请看下面的代码以熟悉该模型。
``` python
def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """
        
    grads = {}
    costs = []                            # to keep track of the cost
    m = X.shape[1]                        # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]
    
    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            
        # Backward propagation.
        assert (lambd == 0 or keep_prob == 1)   # it is possible to use both L2 regularization and dropout, 
                                                # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
```
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/952742b2-4d00-4029-8245-a8ce24c5e2c9)
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/3c155437-c15a-4486-a02d-72616fe8c64b)

训练准确率为94.8%，测试准确率为91.5%。这是基准模型（您将观察正则化对该模型的影响）。运行以下代码绘制模型的决策边界。
```python
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/6ff0718d-cf93-433f-9660-1e8f5cd1f441)

The non-regularized model is obviously overfitting the training set. It is fitting the noisy points! Lets now look at two techniques to reduce overfitting.

<a name='5'></a>
## 5 - L2 Regularization
![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a61bec34-2838-4db3-b1fe-4c00890b2f09)

练习 1 - 使用正则化计算成本
实现compute_cost_with_regularization()，计算式(2)给出的代价。计算∑𝑘∑𝑗所作[𝑙]2𝑘,𝑗时，使用 ：

```
np.sum(np.square(Wl))
```
请注意，您必须先对ᵅ[1]、ᵅ[2]和ᵅ[3]进行计算，然后将三项相加并乘以 1𝑚𝜆2 。
主要代码解释： 其中的lambd = lambda = 入.
```
    L2_regularization_cost = (1 / m) * (lambd / 2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))  # 计算L2正则化成本部分
```
**GRADED FUNCTION: compute_cost_with_regularization**

``` python
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    
    #(≈ 1 lines of code)
    # L2_regularization_cost = 
    # YOUR CODE STARTS HERE
    L2_regularization_cost = (1 / m) * (lambd / 2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))  # 计算L2正则化成本部分

    
    # YOUR CODE ENDS HERE
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost
```
**计算出L2regulation的cost**
```python
A3, t_Y, parameters = compute_cost_with_regularization_test_case()
cost = compute_cost_with_regularization(A3, t_Y, parameters, lambd=0.1)
print("cost = " + str(cost))
compute_cost_with_regularization_test(compute_cost_with_regularization)
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a68e8f46-b685-4fa7-a337-f02000120050)

**当然，由于您改变了成本，您也必须改变反向传播！所有梯度都必须根据新的成本进行计算。**

### Exercise 2 - backward_propagation_with_regularization

Implement the changes needed in backward propagation to take into account regularization. The changes only concern dW1, dW2 and dW3.
For each, you have to add the regularization term's gradient ($\frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m}  W^2) = \frac{\lambda}{m} W$).

###练习2 - 带有正则化的反向传播
实现反向传播中考虑正则化的变化. 这些变化只涉及 dW1、dW2 和 dW3。对于每一个，你都必须添加正则化项的梯度 ($\frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m} W^2) = \frac{\lambda}{m} W$). W$).

**定义进行back_propagation_regularization的函数**


# GRADED FUNCTION: backward_propagation_with_regularization
``` python

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    #(≈ 1 lines of code)
    # dW3 = 1./m * np.dot(dZ3, A2.T) + None
    # YOUR CODE STARTS HERE
    
    dW3 = (1. / m) * np.dot(dZ3, A2.T) + (lambd / m) * W3  # 带有L2正则化的dW3
    # YOUR CODE ENDS HERE
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    #(≈ 1 lines of code)
    # dW2 = 1./m * np.dot(dZ2, A1.T) + None
    # YOUR CODE STARTS HERE
    dW2 = (1. / m) * np.dot(dZ2, A1.T) + (lambd / m) * W2  # 带有L2正则化的dW2
    
    # YOUR CODE ENDS HERE
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    #(≈ 1 lines of code)
    # dW1 = 1./m * np.dot(dZ1, X.T) + None
    # YOUR CODE STARTS HERE
    dW1 = (1. / m) * np.dot(dZ1, X.T) + (lambd / m) * W1  # 带有L2正则化的dW1
    
    # YOUR CODE ENDS HERE
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```

**导入数据计算Regulazation之后的backward_propagation**
```python
t_X, t_Y, cache = backward_propagation_with_regularization_test_case()

grads = backward_propagation_with_regularization(t_X, t_Y, cache, lambd = 0.7)
print ("dW1 = \n"+ str(grads["dW1"]))
print ("dW2 = \n"+ str(grads["dW2"]))
print ("dW3 = \n"+ str(grads["dW3"]))
backward_propagation_with_regularization_test(backward_propagation_with_regularization)
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/0914d3b0-968e-4925-af9f-e6350a2a7357)

Let's now run the model with L2 regularization $(\lambda = 0.7)$. The `model()` function will call: 
- `compute_cost_with_regularization` instead of `compute_cost`
- `backward_propagation_with_regularization` instead of `backward_propagation`


  现在让我们使用L2正则化$(\lambda = 0.7)$运行模型。model()`函数将调用： 
- compute_cost_with_regularization`代替`compute_cost`。
- `backward_propagation_with_regularization`代替`backward_propagation`。
```python  
parameters = model(train_X, train_Y, lambd = 0.7) # 刚才更新了t_X,t_Y,cache
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/98d2374f-9d5a-4a62-a745-8a5c06c4af50)

Congrats, the test set accuracy increased to 93%. You have saved the French football team!

You are not overfitting the training data anymore. Let's plot the decision boundary.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/fab0c51e-7ed7-4066-8d6b-0759e1a0d8ab)


**Observations**:
- The value of $\lambda$ is a hyperparameter that you can tune using a dev set.
- L2 regularization makes your decision boundary smoother. If $\lambda$ is too large, it is also possible to "oversmooth", resulting in a model with high bias.

**What is L2-regularization actually doing?**:

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes. 

<br>
<font color='blue'>
    
**What you should remember:** the implications of L2-regularization on:
- The cost computation:
    - A regularization term is added to the cost.
- The backpropagation function:
    - There are extra terms in the gradients with respect to weight matrices.
- Weights end up smaller ("weight decay"): 
    - Weights are pushed to smaller values.


  观察结果：

𝜆的值是一个超参数，您可以使用偏差集进行调整。
L2正则化使决策边界更加平滑。如果𝜆过大，也有可能 "过度平滑"，导致模型偏差过大。
L2-正则化的实际作用是什么？

L2-正则化基于这样一个假设：权重小的模型比权重大的模型简单。因此，通过惩罚代价函数中权重的平方值，可以使所有权重值变小。大权重的代价太高！这将导致一个更平滑的模型，其中输出随着输入的变化而变化得更慢。


**您应该记住的：L2-正则化的影响：**
- 代价计算
- 正则化项被添加到代价中。
- 反向传播函数：
- 在权重矩阵的梯度中有额外的项。
- 权重最终变小（"权重衰减"）：
- 权重被推至更小的值。



    



# Regularization

请注意，正则化会损害训练集的性能！这是因为它限制了网络过拟合训练集的能力。但是，由于正则化最终会提高测试精度，因此它对您的系统是有帮助的。

：

- 正则化将帮助您减少过度拟合。
- 正则化将降低权重值。
- L2正则化和Dropout是两种非常有效的正则化技术。

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


**使用Dropout来进行regularization**

Finally, **dropout** is a widely used regularization technique that is specific to deep learning. It randomly shuts down some neurons in each iteration. Watch these two videos to see what this means!

最后，**dropout**是一种广泛使用的正则化技术，专门用于深度学习。它在每次迭代中随机关闭一些神经元。请观看这两段视频，了解这意味着什么！

Dropout算法让每一次iteration中丢失的神经元都是随机的.

https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/fb550c6d-dead-4725-9e77-045059024910
Figure 2 : Drop-out on the second hidden layer.
At each iteration, you shut down (= set to zero) each neuron of a layer with probability  1−𝑘𝑒𝑒𝑝_𝑝𝑟𝑜𝑏  or keep it with probability  𝑘𝑒𝑒𝑝_𝑝𝑟𝑜𝑏  (50% here). The dropped neurons don't contribute to the training in both the forward and backward propagations of the iteration.
图 2 : 第二层隐藏神经元的退出。
在每次迭代中，以概率1-𝑘𝑒𝑝_𝑝𝑟𝑜𝑏或以概率𝑘𝑒𝑝_𝑝𝑟𝑜𝑏（此处为50%）保留一层中的每个神经元。被删除的神经元在迭代的前向和后向传播中对训练没有贡献。

https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/5471a1f4-30ec-45f3-9d45-99b7543b3acc

Figure 3: Drop-out on the first and third hidden layers.
1𝑠𝑡  layer: we shut down on average 40% of the neurons.  3𝑟𝑑  layer: we shut down on average 20% of the neurons.

图3：第一和第三隐藏层的神经元丢失。

1𝑠𝑡层：我们平均关闭了40%的神经元。 3𝑟𝑑层：我们平均关闭了20%的神经元。

当你关闭一些神经元时，你实际上修改了你的模型。Drop-out背后的理念是，在每次迭代中，您都要训练一个不同的模型，该模型只使用神经元的一个子集。
通过停用，神经元对其他特定神经元的激活变得不那么敏感，因为其他神经元随时可能被关闭。


<a name='6-1'></a>
### 6.1 - Forward Propagation with Dropout

<a name='ex-3'></a>
### Exercise 3 - forward_propagation_with_dropout

Implement the forward propagation with dropout. You are using a 3 layer neural network, and will add dropout to the first and second hidden layers. We will not apply dropout to the input layer or output layer. 

**Instructions**:
You would like to shut down some neurons in the first and second layers. To do that, you are going to carry out 4 Steps:
1. In lecture, we dicussed creating a variable $d^{[1]}$ with the same shape as $a^{[1]}$ using `np.random.rand()` to randomly get numbers between 0 and 1. Here, you will use a vectorized implementation, so create a random matrix $D^{[1]} = [d^{[1](1)} d^{[1](2)} ... d^{[1](m)}] $ of the same dimension as $A^{[1]}$.
2. Set each entry of $D^{[1]}$ to be 1 with probability (`keep_prob`), and 0 otherwise.

**Hint:** Let's say that keep_prob = 0.8, which means that we want to keep about 80% of the neurons and drop out about 20% of them.  We want to generate a vector that has 1's and 0's, where about 80% of them are 1 and about 20% are 0.
This python statement:  
`X = (X < keep_prob).astype(int)`  

is conceptually the same as this if-else statement (for the simple case of a one-dimensional array) :

```
for i,v in enumerate(x):
    if v < keep_prob:
        x[i] = 1
    else: # v >= keep_prob
        x[i] = 0
```
Note that the `X = (X < keep_prob).astype(int)` works with multi-dimensional arrays, and the resulting output preserves the dimensions of the input array.

Also note that without using `.astype(int)`, the result is an array of booleans `True` and `False`, which Python automatically converts to 1 and 0 if we multiply it with numbers.  (However, it's better practice to convert data into the data type that we intend, so try using `.astype(int)`.)

3. Set $A^{[1]}$ to $A^{[1]} * D^{[1]}$. (You are shutting down some neurons). You can think of $D^{[1]}$ as a mask, so that when it is multiplied with another matrix, it shuts down some of the values.
4. Divide $A^{[1]}$ by `keep_prob`. By doing this you are assuring that the result of the cost will still have the same expected value as without drop-out. (This technique is also called inverted dropout.)

练习 3 - 带滤波的前向传播
实现带滤波的前向传播。您将使用一个三层神经网络，并在第一层和第二层隐藏层添加滤波。我们不会在输入层和输出层添加滤波。

说明： 您希望关闭第一层和第二层的一些神经元。为此，您需要执行4个步骤：
**注意，dropout只处理hidden层，不处理output层**
在讲座中，我们讨论了使用np.random.rand()创建一个与𝑎[1]形状相同的变量𝑑[1]，**随机获取0到1之间的数字**。在这里，您将使用矢量化实现，因此创建一个与 𝐴[1]相同维度的随机矩阵 𝐷[1]=[𝑑[1](1)𝑑[1](2)...𝑑[1](𝑚)] 。
将𝐷[1]中的每个条目设置为 1，概率为 (keep_prob)，否则为 0。
提示：假设 keep_prob = 0.8，这意味着我们希望保留大约 80% 的神经元，放弃大约 20% 的神经元。我们希望生成一个有1和0的向量，其中大约80%是1，大约20%是0：
X = (X < keep_prob).astype(int)

在概念上与if-else语句相同（对于一维数组的简单情况）：
```
for i,v in enumerate(x)：
    if v < keep_prob：
        x[i] = 1
    else： # v >= keep_prob
        x[i] = 0
```
注意X = (X < keep_prob).astype(int)对多维数组有效，输出结果保留了输入数组的维数。

还要注意的是，如果不使用 .astype(int)，结果将是一个布尔数组 True 和 False，如果我们将其与数字相乘，Python 会自动将其转换为 1 和 0。(然而，更好的做法是将数据转换成我们想要的数据类型，所以尝试使用 .astype(int))。

将 𝐴[1] 设为 𝐴[1]∗𝐷[1] 。(您正在关闭一些神经元）。您可以将 𝐷[1]视为一个掩码，当它与另一个矩阵相乘时，它将关闭某些值。
用 keep_prob 除以 𝐴[1]。通过这样做，您可以确保代价的结果仍然具有与没有丢弃时相同的期望值。(这种技术也被称为反向滤波）。
# GRADED FUNCTION: forward_propagation_with_dropout
```python
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    #(≈ 4 lines of code)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
    # D1 =                                           # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    # D1 =                                           # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    # A1 =                                           # Step 3: shut down some neurons of A1
    # A1 =                                           # Step 4: scale the value of neurons that haven't been shut down
    # YOUR CODE STARTS HERE
    D1 = np.random.rand(A1.shape[0], A1.shape[1])  # Step 1: initialize matrix D1
    D1 = (D1 < keep_prob).astype(int)  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1 * D1  # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down
    
    # YOUR CODE ENDS HERE
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    #(≈ 4 lines of code)
    # D2 =                                           # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    # D2 =                                           # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    # A2 =                                           # Step 3: shut down some neurons of A2
    # A2 =                                           # Step 4: scale the value of neurons that haven't been shut down
    # YOUR CODE STARTS HERE
    D2 = np.random.rand(A2.shape[0], A2.shape[1])  # Step 1: initialize matrix D2
    D2 = (D2 < keep_prob).astype(int)  # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2 * D2  # Step 3: shut down some neurons of A2
    A2 = A2 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down
    
    # YOUR CODE ENDS HERE
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/aec9fee9-def1-4975-9122-af22f89c1e77)


<a name='6-2'></a>
### 6.2 - Backward Propagation with Dropout

<a name='ex-4'></a>
### Exercise 4 - backward_propagation_with_dropout
Implement the backward propagation with dropout. As before, you are training a 3 layer network. Add dropout to the first and second hidden layers, using the masks $D^{[1]}$ and $D^{[2]}$ stored in the cache. 



**Instruction**:
Backpropagation with dropout is actually quite easy. You will have to carry out 2 Steps:
1. You had previously shut down some neurons during forward propagation, by applying a mask $D^{[1]}$ to `A1`. In backpropagation, you will have to shut down the same neurons, by reapplying the same mask $D^{[1]}$ to `dA1`. 
2. During forward propagation, you had divided `A1` by `keep_prob`. In backpropagation, you'll therefore have to divide `dA1` by `keep_prob` again (the calculus interpretation is that if $A^{[1]}$ is scaled by `keep_prob`, then its derivative $dA^{[1]}$ is also scaled by the same `keep_prob`).

6.2 - 带滤波的反向传播

练习 4 - 带滤波的反向传播

实现带 dropout 的反向传播。和以前一样，我们训练一个三层网络。使用缓存中的掩码𝐷[1]和𝐷[2]，在第一层和第二层隐藏层中添加滤波。

指令： 带 dropout 的反向传播实际上非常简单。您需要执行 2 个步骤：

- 在前向传播过程中，通过对 A1 应用掩码 𝐷[1]，您已经关闭了一些神经元。在反向传播过程中，您需要关闭相同的神经元，方法是在 dA1 上重新应用相同的掩码𝐷[1]。
- 在前向传播过程中，您将 A1 除以 keep_prob。在反向传播中，您必须再次用 keep_prob 除以 dA1（微积分的解释是，如果 𝐴[1] 被 keep_prob 缩放，那么它的导数 𝑑𝐴[1]也被同样的 keep_prob 缩放）。
- **很简单，也是把随机出来的数组（0到1之间的）直接和算出来的dA相乘，再除以keep_prob来**

# GRADED FUNCTION: backward_propagation_with_dropout
```python
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    #(≈ 2 lines of code)
    # dA2 =                # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    # dA2 =                # Step 2: Scale the value of neurons that haven't been shut down
    # YOUR CODE STARTS HERE
    dA2 = np.multiply(dA2, D2)
    dA2 = dA2 / keep_prob  # Scale the value of neurons that haven't been shut down
    
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    
    # YOUR CODE ENDS HERE
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    #(≈ 2 lines of code)
    # dA1 =                # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    # dA1 =                # Step 2: Scale the value of neurons that haven't been shut down
    # YOUR CODE STARTS HERE现在让我们运行带 dropout 的模型（`keep_prob = 0.86`）。这意味着在每次迭代时，以14%的概率关闭第1层和第2层的每个神经元。函数 `model()` 现在将调用：
- 前向_propagation_with_dropout"，而不是 "前向_propagation"。
- `backward_propagation_with_dropout`代替`backward_propagation`。
    dA1 = np.multiply(dA1,D1)
    dA1 = dA1/ keep_prob
    # YOUR CODE ENDS HERE
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/ed1ee673-3fd2-47a0-9e63-b4a9c0eca2af)

Let's now run the model with dropout (`keep_prob = 0.86`). It means at every iteration you shut down each neurons of layer 1 and 2 with 14% probability. The function `model()` will now call:
- `forward_propagation_with_dropout` instead of `forward_propagation`.
- `backward_propagation_with_dropout` instead of `backward_propagation`.


现在让我们运行带 dropout 的模型（`keep_prob = 0.86`）。这意味着在每次迭代时，以14%的概率关闭第1层和第2层的每个神经元。函数 `model()` 现在将调用：
- 前向_propagation_with_dropout"，而不是 "前向_propagation"。
- `backward_propagation_with_dropout`代替`backward_propagation`。
 ```
parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/0fbd6aa3-00aa-452b-942a-dc1909c4ae38)

Dropout运行良好！测试准确率再次提高（达到 95%）！您的模型没有过度拟合训练集，并且在测试集上表现出色。法国足球队将永远感谢您！

运行下面的代码绘制决策边界。
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/59298adb-8675-439b-9150-2d96256d52e7)


**注意**：
- 使用dropout时的一个**常见错误是在训练和测试中都使用它。您应该只在训练中使用dropout（随机剔除节点）。
- 像[TensorFlow](https://www.tensorflow.org/api_docs/python/tf/nn/dropout)、[PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/nn/Dropout_en.html#dropout)、[Keras](https://keras.io/api/layers/regularization_layers/dropout/)或[caffe](https://caffe.berkeleyvision.org/doxygen/classcaffe_1_1DropoutLayer.html)这样的深度学习框架都有dropout层实现。不要紧张--您很快就会学会其中的一些框架。

<font color='blue'>
    
**关于Dropout，您需要记住的是：**
- Dropout是一种正则化技术。
- 你只能在训练时使用dropout。在测试时不要使用dropout（随机消除节点）。
- 在前向和后向传播过程中都要使用dropout。
- 在训练期间，将每个 dropout 层除以 keep_prob，以保持激活的期望值相同。例如，如果keep_prob为0.5，那么我们将平均关闭一半的节点，因此输出将按0.5的比例缩放，因为只有剩余的一半节点对求解有贡献。因此，现在的输出具有相同的期望值。您可以检查一下，即使 keep_prob 的值不是 0.5，这个方法也是有效的。 

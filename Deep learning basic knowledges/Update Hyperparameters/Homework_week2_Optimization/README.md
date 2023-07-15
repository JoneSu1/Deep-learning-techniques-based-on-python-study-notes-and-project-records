speed up learning and perhaps even get you to a better final value for the cost function. 
Having a good optimization algorithm can be the difference between waiting days vs. 
just a few hours to get a good result.

## Table of Contents
- [1- Packages](#1)
- [2 - Gradient Descent](#2)
    - [Exercise 1 - update_parameters_with_gd](#ex-1)
- [3 - Mini-Batch Gradient Descent](#3)
    - [Exercise 2 - random_mini_batches](#ex-2)
- [4 - Momentum](#4)
    - [Exercise 3 - initialize_velocity](#ex-3)
    - [Exercise 4 - update_parameters_with_momentum](#ex-4)
- [5 - Adam](#5)
    - [Exercise 5 - initialize_adam](#ex-5)
    - [Exercise 6 - update_parameters_with_adam](#ex-6)
- [6 - Model with different Optimization algorithms](#6) 
    - [6.1 - Mini-Batch Gradient Descent](#6-1)
    - [6.2 - Mini-Batch Gradient Descent with Momentum](#6-2)
    - [6.3 - Mini-Batch with Adam](#6-3)
    - [6.4 - Summary](#6-4)
- [7 - Learning Rate Decay and Scheduling](#7)
    - [7.1 - Decay on every iteration](#7-1)
        - [Exercise 7 - update_lr](#ex-7)
    - [7.2 - Fixed Interval Scheduling](#7-2)
        - [Exercise 8 - schedule_lr_decay](#ex-8)
    - [7.3 - Using Learning Rate Decay for each Optimization Method](#7-3)
        - [7.3.1 - Gradient Descent with Learning Rate Decay](#7-3-1)
        - [7.3.2 - Gradient Descent with Momentum and Learning Rate Decay](#7-3-2)
        - [7.3.3 - Adam with Learning Rate Decay](#7-3-3)
    - [7.4 - Achieving similar performance with different methods](#7-4)
 
  <a name='1'></a>
## 1- Packages
``` python

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils_v1a import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils_v1a import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from copy import deepcopy
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
## 2 - Gradient Descent

A simple optimization method in machine learning is gradient descent (GD). When you take gradient steps with respect to all $m$ examples on each step,
it is also called Batch Gradient Descent. 

<a name='ex-1'></a>
### Exercise 1 - update_parameters_with_gd

Implement the gradient descent update rule. The  gradient descent rule is, for $l = 1, ..., L$: 
$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{1}$$
$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{2}$$

where L is the number of layers and $\alpha$ is the learning rate. 
All parameters should be stored in the `parameters` dictionary. Note that the iterator `l` starts at 1
in the `for` loop as the first parameters are $W^{[1]}$ and $b^{[1]}$. 

# GRADED FUNCTION: update_parameters_with_gd
``` python
def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    L = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(1, L + 1):
        # (approx. 2 lines)
        # parameters["W" + str(l)] =  
        # parameters["b" + str(l)] = 
        # YOUR CODE STARTS HERE
        parameters["W" + str(l)] =parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] =parameters["b" + str(l)] - learning_rate * grads["db"+ str(l)]
        # YOUR CODE ENDS HERE
    return parameters
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/91603b65-5586-4719-9f6f-c5f0019c3078)

A variant of this is Stochastic Gradient Descent (SGD), which is equivalent to mini-batch gradient descent, where each mini-batch has just 1 example. The update rule that you have just implemented does not change. What changes is that you would be computing gradients on just one training example at a time, rather than on the whole training set. The code examples below illustrate the difference between stochastic gradient descent and (batch) gradient descent.

其变种是随机梯度下降法（SGD），相当于迷你批次梯度下降法，每个迷你批次只有一个实例。您刚刚实现的更新规则并没有改变。发生变化的是，您每次只在一个训练实例上计算梯度，而不是在整个训练集上计算梯度。下面的代码示例说明了随机梯度下降和（批量）梯度下降的区别。

- **(Batch) Gradient Descent**:

``` python
X = data_input
Y = labels
m = X.shape[1]  # Number of training examples
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    # Forward propagation
    a, caches = forward_propagation(X, parameters)
    # Compute cost
    cost_total = compute_cost(a, Y)  # Cost for m training examples
    # Backward propagation
    grads = backward_propagation(a, caches, parameters)
    # Update parameters
    parameters = update_parameters(parameters, grads)
    # Compute average cost
    cost_avg = cost_total / m
        
```
#差别就是在Stochastic gradient descent里面还是用了第二个for loop J，j in range(0,分出的batch数目).

- **Stochastic Gradient Descent**:

```python
X = data_input
Y = labels
m = X.shape[1]  # Number of training examples
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    cost_total = 0
    for j in range(0, m):
        # Forward propagation
        a, caches = forward_propagation(X[:,j], parameters)
        # Compute cost
        cost_total += compute_cost(a, Y[:,j])  # Cost for one training example
        # Backward propagation
        grads = backward_propagation(a, caches, parameters)
        # Update parameters
        parameters = update_parameters(parameters, grads)
    # Compute average cost
    cost_avg = cost_total / m
```

In Stochastic Gradient Descent, you use only 1 training example before updating the gradients. When the training set is large, SGD can be faster. But the parameters will "oscillate（震动）" toward the minimum rather than converge smoothly. Here's what that looks like:

![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/7d79f6c3-b871-457b-a0cb-ccfdbbf1abae)

**Note** also that implementing SGD requires 3 for-loops in total:
1. Over the number of iterations
2. Over the $m$ training examples
3. Over the layers (to update all parameters, from $(W^{[1]},b^{[1]})$ to $(W^{[L]},b^{[L]})$)

In practice, you'll often get faster results if you don't use the entire training set, or just one training example, to perform each update. Mini-batch gradient descent uses an intermediate number of examples for each step. With mini-batch gradient descent, you loop over the mini-batches instead of looping over individual training examples.

在实践中，如果不使用整个训练集，或只使用一个训练示例来执行每次更新，通常会获得更快的结果。迷你批次梯度下降法每一步都使用中间数量的示例。使用迷你批次梯度下降法，您可以循环使用迷你批次，而不是循环使用单个训练示例。

![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/5c16155d-dee0-475e-b0b6-ac2bb937f93c)


<a name='3'></a>
## 3 - Mini-Batch Gradient Descent

Now you'll build some mini-batches from the training set (X, Y).

There are two steps:

第一步就是把数据随机的分到mini——batch中
第二步就是，把洗牌后的数据按大小（64）分成mini-bath

- **Shuffle**: Create a shuffled version of the training set (X, Y) as shown below. Each column of X and Y represents a training example. Note that the random shuffling is done synchronously between X and Y. Such that after the shuffling the $i^{th}$ column of X is the example corresponding to the $i^{th}$ label in Y. The shuffling step ensures that examples will be split randomly into different mini-batches. 


![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/dc16967c-cc47-4b5d-b313-753db3cc4dd3)

- **Partition**: Partition the shuffled (X, Y) into mini-batches of size `mini_batch_size` (here 64). Note that the number of training examples is not always divisible by `mini_batch_size`. The last mini batch might be smaller, but you don't need to worry about this. When the final mini-batch is smaller than the full `mini_batch_size`, it will look like this:

-  **分区**： 将洗牌后的(X, Y)分成大小为`mini_batch_size`（此处为64）的迷你批。请注意，训练实例的数量并不总是可以被 "mini_batch_size "整除。最后的迷你批次可能会更小，但无需担心。当最后的迷你批次小于全部的`mini_batch_size`时，它将看起来像这样：
  
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/430151c2-72ce-40bd-ad56-67b52c322af8)

<a name='ex-2'></a>
### Exercise 2 - random_mini_batches

Implement `random_mini_batches`. The shuffling part has already been coded for you! To help with the partitioning step, you've been provided the following code that selects the indexes for the $1^{st}$ and $2^{nd}$ mini-batches:
```python
first_mini_batch_X = shuffled_X[:, 0 : mini_batch_size]
second_mini_batch_X = shuffled_X[:, mini_batch_size : 2 * mini_batch_size]
...
```

Note that the last mini-batch might end up smaller than `mini_batch_size=64`. Let $\lfloor s \rfloor$ represents $s$ rounded down to the nearest integer (this is `math.floor(s)` in Python). If the total number of examples is not a multiple of `mini_batch_size=64` then there will be $\left\lfloor \frac{m}{mini\_batch\_size}\right\rfloor$ mini-batches with a full 64 examples, and the number of examples in the final mini-batch will be![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/17460df1-9fc4-4b88-8843-315acdf77cc6)


**Hint:**

$$mini\_batch\_X = shuffled\_X[:, i : j]$$ 

Think of a way in which you can use the for loop variable `k` help you increment `i` and `j` in multiples of mini_batch_size.

As an example, if you want to increment in multiples of 3, you could the following:

```python
n = 3
for k in (0 , 5):
    print(k * n)
```


在代码中，k 是循环变量，它表示当前迭代的完整小批次的索引。索引从0开始，所以 (k + 1) 表示下一个完整小批次的索引。

在处理完整小批次时，我们需要从 shuffled_X 和 shuffled_Y 中提取一段连续的数据作为当前小批次。这段数据的起始索引是 k * mini_batch_size，结束索引是 (k + 1) * mini_batch_size。因此，我们需要使用 (k + 1) 来计算结束索引，以确保提取的数据是连续的、不重叠的。

举个例子，如果 mini_batch_size 是 64，而 k 的值是 0，则 (k + 1) * mini_batch_size 就是 64，意味着我们提取的数据范围是从索引 0 到 63（共计 64 个元素），即一个完整的小批次。
```python
import math
import numpy as np

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))  # 随机打乱样本顺序
    shuffled_X = X[:, permutation]  # 根据打乱的索引重新排列X
    shuffled_Y = Y[:, permutation].reshape((1, m))  # 根据打乱的索引重新排列Y

    inc = mini_batch_size

    # Step 2 - Partition (shuffled_X, shuffled_Y).
    # Cases with a complete mini batch size only i.e each of 64 examples.
    num_complete_minibatches = math.floor(m / mini_batch_size)  # 计算完整小批次的数量
    for k in range(0, num_complete_minibatches):
        # 提取完整的小批次
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
    if m % mini_batch_size != 0:
        # 提取最后一个不完整的小批次
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
```
```python
np.random.seed(1)  # 设置随机种子为1
mini_batch_size = 64  # 小批次大小
nx = 12288  # 输入特征的数量
m = 148  # 样本数量

# 创建输入数据 X，大小为 (nx, m)
X = np.array([x for x in range(nx * m)]).reshape((m, nx)).T

# 创建标签数据 Y，大小为 (1, m)
Y = np.random.randn(1, m) < 0.5

# 生成随机小批次
mini_batches = random_mini_batches(X, Y, mini_batch_size)

# 计算生成的小批次数量
n_batches = len(mini_batches)

# 检查生成的小批次数量是否正确
assert n_batches == math.ceil(m / mini_batch_size), f"Wrong number of mini batches. {n_batches} != {math.ceil(m / mini_batch_size)}"

# 检查每个小批次的形状和数值
for k in range(n_batches - 1):
    assert mini_batches[k][0].shape == (nx, mini_batch_size), f"Wrong shape in {k} mini batch for X"
    assert mini_batches[k][1].shape == (1, mini_batch_size), f"Wrong shape in {k} mini batch for Y"
    assert np.sum(np.sum(mini_batches[k][0] - mini_batches[k][0][0], axis=0)) == ((nx * (nx - 1) / 2 ) * mini_batch_size), "Wrong values. It happens if the order of X rows(features) changes"

# 检查最后一个小批次的形状
if m % mini_batch_size > 0:
    assert mini_batches[n_batches - 1][0].shape == (nx, m % mini_batch_size), f"Wrong shape in the last minibatch. {mini_batches[n_batches - 1][0].shape} != {(nx, m % mini_batch_size)}"

# 检查特定索引处的数值是否正确
assert np.allclose(mini_batches[0][0][0][0:3], [294912,  86016, 454656]), "Wrong values. Check the indexes used to form the mini batches"
assert np.allclose(mini_batches[-1][0][-1][0:3], [1425407, 1769471, 897023]), "Wrong values. Check the indexes used to form the mini batches"

print("\033[92mAll tests passed!")
```

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/1250c8a1-7a66-4194-b094-c9668324faf8)

<font color='blue'>
    
**您应该记住**：
- 洗牌和分区是建立迷你批所需的两个步骤
- 通常选择2的幂次作为迷你批的大小，例如16、32、64、128。


<a name='4'></a>
## 4 - Momentum

Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations. 

Momentum takes into account the past gradients to smooth out the update. The 'direction' of the previous gradients is stored in the variable $v$. Formally, this will be the exponentially weighted average of the gradient on previous steps. You can also think of $v$ as the "velocity" of a ball rolling downhill, building up speed (and momentum) according to the direction of the gradient/slope of the hill. 

由于微型批量梯度下降算法只在看到一个子集的例子后进行参数更新，因此更新的方向有一定的偏差，所以微型批量梯度下降算法的收敛路径会出现 "振荡"。使用动量可以减少这些振荡。

动量考虑了过去的梯度来平滑更新。之前梯度的 "方向 "存储在变量𝑣中。从形式上看，这是前几步梯度的指数加权平均值。您也可以将 𝑣 看作是下坡滚动的球的 "速度"，根据坡度/斜率的方向增加速度（和动量）。
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/db73d9ec-766c-474e-a404-1b64d48c2f16)

<caption><center> <u><font color='purple'><b>Figure 3</b> </u><font color='purple'>: The red arrows show the direction taken by one step of mini-batch gradient descent with momentum. The blue points show the direction of the gradient (with respect to the current mini-batch) on each step. Rather than just following the gradient, the gradient is allowed to influence $v$ and then take a step in the direction of $v$.<br> <font color='black'> </center>

图 3 : 红色箭头表示带动量的迷你批次梯度下降的一步方向。蓝色点表示每一步的梯度方向（相对于当前迷你批次）。图 3：红色箭头表示带动量的小批量梯度下降过程中的每一步，
蓝色点表示每一步的梯度方向（相对于当前的小批量）。

<a name='ex-3'></a>    
### Exercise 3 - initialize_velocity
Initialize the velocity. The velocity, $v$, is a python dictionary that needs to be initialized with arrays of zeros. Its keys are the same as those in the `grads` dictionary, that is:
for $l =1,...,L$:
初始化速度 速度 𝑣 是一个 python 字典，需要用零数组进行初始化。其键值与grads字典中的键值相同，即：对于𝑙=1,...,𝐿 ：
```python
v["dW" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["W" + str(l)])
v["db" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["b" + str(l)])
```
**Note** that the iterator l starts at 1 in the for loop as the first parameters are v["dW1"] and v["db1"] (that's a "one" on the superscript).
**注意**在for循环中迭代器l从1开始，因为第一个参数是v["dW1"]和v["db1"]（上标是 "一"）。

np.zeros_like()是NumPy库中的一个函数，用于创建一个与给定数组具有相同形状的零数组。

具体而言，np.zeros_like(arr)函数将返回一个与数组arr具有相同形状的零数组。该函数的返回值是一个新的NumPy数组，其中的元素都被初始化为零。

```python
def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2  # 神经网络中的层数
    v = {}
    
    # 初始化速度
    for l in range(1, L + 1):
        # 初始化v["dW" + str(l)]和v["db" + str(l)]为零数组
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        
    return v
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/02882220-4831-4320-8d69-d13db67fe2e9)

<a name='ex-4'></a>   
### Exercise 4 - update_parameters_with_momentum

Now, implement the parameters update with momentum. The momentum update rule is, for $l = 1, ..., L$: 

$$ \begin{cases}
v_{dW^{[l]}} = \beta v_{dW^{[l]}} + (1 - \beta) dW^{[l]} \\
W^{[l]} = W^{[l]} - \alpha v_{dW^{[l]}}
\end{cases}\tag{3}$$

$$\begin{cases}
v_{db^{[l]}} = \beta v_{db^{[l]}} + (1 - \beta) db^{[l]} \\
b^{[l]} = b^{[l]} - \alpha v_{db^{[l]}} 
\end{cases}\tag{4}$$

where L is the number of layers, $\beta$ is the momentum and $\alpha$ is the learning rate. All parameters should be stored in the `parameters` dictionary.  Note that the iterator `l` starts at 1 in the `for` loop as the first parameters are $W^{[1]}$ and $b^{[1]}$ (that's a "one" on the superscript).

其中，L 是层数， 𝛼 是动量，𝛼 是学习率。所有参数都存储在参数字典中。请注意，在for循环中，迭代器l从1开始，因为第一个参数是𝑏[1]和𝑏[1]（上标是 "1"）。
```python
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2  # 神经网络中的层数
    
    # 对每个参数进行动量更新
    for l in range(1, L + 1):
        
        # 计算速度
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
        
        # 更新参数
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v["db" + str(l)]
        
    return parameters, v
```
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/b2d28a5d-4ce3-4b24-b457-7b4d997d2636)


v["dW" + str(l)]的计算公式为 beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
代码中计算 v["dW" + str(l)] 的公式确实是使用了历史速度 v["dW" + str(l)] 和当前梯度 grads["dW" + str(l)]。这是为了实现动量方法的更新规则。

动量方法中的速度 v["dW" + str(l)] 可以看作是参数更新的一个积累量，用于记录历史梯度的影响。它在每次迭代中都会被更新，而不仅仅依赖于当前的梯度。通过结合历史速度和当前梯度，动量方法能够在参数更新中保持一定的惯性，从而加速收敛并平滑优化路径。

因此，正确的计算公式为 v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]，其中 v["dW" + str(l)] 表示当前速度，beta 是动量超参数，grads["dW" + str(l)] 是当前梯度。

在代码中，您可以看到这个公式的实际应用，即更新 v["dW" + str(l)] 的数值。这样，在接下来的步骤中，可以使用更新后的速度来更新参数。


**Note that**:
- The velocity is initialized with zeros. So the algorithm will take a few iterations to "build up" velocity and start to take bigger steps.
- If $\beta = 0$, then this just becomes standard gradient descent without momentum. 

**How do you choose $\beta$?**

- The larger the momentum $\beta$ is, the smoother the update, because it takes the past gradients into account more. But if $\beta$ is too big, it could also smooth out the updates too much. 
- Common values for $\beta$ range from 0.8 to 0.999. If you don't feel inclined to tune this, $\beta = 0.9$ is often a reasonable default. 
- Tuning the optimal $\beta$ for your model might require trying several values to see what works best in terms of reducing the value of the cost function $J$.
  

注意

速度初始化为零。因此，算法需要经过几次迭代来 "建立 "速度，并开始采取更大的步长。
如果 φ=0 ，那么就变成了没有动量的标准梯度下降算法。
如何选择 仸？

ν越大，更新越平滑，因为它更多地考虑了过去的梯度。但是，如果 ν φ 过大，也会使更新过于平滑。
常用的 ν 值范围在 0.8 到 0.999 之间。如果您不想调整这个值，通常默认值为0.9。
为您的模型调整最佳的 𝐽 可能需要尝试几个值，看看哪个值在降低成本函数 𝐽 值方面效果最好。


您应该记住的

动量将过去的梯度考虑在内，以平滑梯度下降的步骤。它可以应用于批量梯度下降、迷你批量梯度下降或随机梯度下降。

您必须调整动量超参数 𝛼 和学习率 𝛼 。

<a name='5'></a>   
## 5 - Adam

Adam is one of the most effective optimization algorithms for training neural networks. It combines ideas from RMSProp (described in lecture) and Momentum. 

**How does Adam work?**
1. It calculates an exponentially weighted average of past gradients, and stores it in variables $v$ (before bias correction) and $v^{corrected}$ (with bias correction). 
2. It calculates an exponentially weighted average of the squares of the past gradients, and  stores it in variables $s$ (before bias correction) and $s^{corrected}$ (with bias correction). 
3. It updates parameters in a direction based on combining information from "1" and "2".
 
**Adam 是如何工作的？**
1. 它计算过去梯度的指数加权平均值，并将其存储在变量𝑣（偏差修正前）和𝑣𝑐𝑜𝑟𝑒𝑐𝑡𝑑（偏差修正后）中。
2. 计算过去梯度平方的指数加权平均值，并将其存储在变量𝑠（偏差修正前）和 𝑠𝑐𝑜𝑟𝑒𝑡𝑒𝑑（偏差修正后）中。

3. 根据 "1 "和 "2 "的信息更新参数方向。

The update rule is, for $l = 1, ..., L$: 
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d30b3241-2ae1-406c-a3d6-2f934dc399b0)


where:
- t counts the number of steps taken of Adam 
- L is the number of layers
- $\beta_1$ and $\beta_2$ are hyperparameters that control the two exponentially weighted averages. 
- $\alpha$ is the learning rate
- $\varepsilon$ is a very small number to avoid dividing by zero

As usual, all parameters are stored in the `parameters` dictionary  

- t为Adam的步数
- L是层数
- 𝛼 和 𝛼2 是超参数，用于控制两个指数加权平均值。

- 𝛼是学习率
- 𝜀是一个非常小的数字，以避免除以0。
- 像往常一样，所有参数都存储在参数字典中。
 
<a name='ex-5'></a>   
### Exercise 5 - initialize_adam

Initialize the Adam variables $v, s$ which keep track of the past information.

**Instruction**: The variables $v, s$ are python dictionaries that need to be initialized with arrays of zeros. Their keys are the same as for `grads`, that is:
for $l = 1, ..., L$:
```python
v["dW" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["W" + str(l)])
v["db" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["b" + str(l)])
s["dW" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["W" + str(l)])
s["db" + str(l)] = ... #(numpy array of zeros with the same shape as parameters["b" + str(l)])

```

# GRADED FUNCTION: initialize_adam
```python
def initialize_adam(parameters):
    """
    初始化v和s，它们是两个Python字典：
                - 键: "dW1", "db1", ..., "dWL", "dbL" 
                - 值: 形状与相应梯度/参数相同的零数组
    
    参数：
    parameters -- 包含参数的Python字典。
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    返回: 
    v -- 包含梯度的指数加权平均值的Python字典。初始化为零。
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- 包含平方梯度的指数加权平均值的Python字典。初始化为零。
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...
    """

    L = len(parameters) // 2  # 神经网络中的层数
    v = {}
    s = {}
    
    # 初始化v和s。输入: "parameters"。输出: "v, s"。
    for l in range(1, L + 1):
        # v["dW" + str(l)] = ...
        # v["db" + str(l)] = ...
        # s["dW" + str(l)] = ...
        # s["db" + str(l)] = ...
        # YOUR CODE STARTS HERE
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])  # 用零初始化v["dW" + str(l)]
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])  # 用零初始化v["db" + str(l)]
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])  # 用零初始化s["dW" + str(l)]
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])  # 用零初始化s["db" + str(l)]
        # YOUR CODE ENDS HERE
    
    return v, s
```
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c3a97500-2ac1-46fb-acc2-6ffb1cb51c7e)

<a name='ex-6'></a>   
### Exercise 6 - update_parameters_with_adam

Now, implement the parameters update with Adam. Recall the general update rule is, for $l = 1, ..., L$: 

![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d756e7bf-1f61-49c1-8429-0d065420f9e5)


np.power() 函数是NumPy中用于求幂的函数。它可以用来计算一个数组中元素的幂。

函数的语法如下：
``` PYTHON
np.power(base, exponent)

```

```PYTHON
# GRADED FUNCTION: update_parameters_with_adam

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    使用Adam更新参数
    
    参数：
    parameters -- 包含参数的Python字典:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- 包含每个参数的梯度的Python字典:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam变量，第一个梯度的移动平均值，Python字典
    s -- Adam变量，平方梯度的移动平均值，Python字典
    t -- Adam变量，步数计数
    learning_rate -- 学习率，标量
    beta1 -- 第一矩估计的指数衰减超参数
    beta2 -- 第二矩估计的指数衰减超参数
    epsilon -- 防止Adam更新中除以零的超参数

    返回:
    parameters -- 包含更新后参数的Python字典
    v -- Adam变量，第一个梯度的移动平均值，Python字典
    s -- Adam变量，平方梯度的移动平均值，Python字典
    """
    
    L = len(parameters) // 2                 # 神经网络中的层数
    v_corrected = {}                         # 初始化第一个矩估计的修正值，Python字典
    s_corrected = {}                         # 初始化第二个矩估计的修正值，Python字典
    
    # 对所有参数执行Adam更新
    for l in range(1, L + 1):
        # 梯度的移动平均值。输入: "v, grads, beta1"。输出: "v"。
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

        # 计算偏差修正的第一个矩估计。输入: "v, beta1, t"。输出: "v_corrected"。
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))

        # 平方梯度的移动平均值。输入: "s, grads, beta2"。输出: "s"。
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.power(grads["dW" + str(l)], 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.power(grads["db" + str(l)], 2)

        # 计算偏差修正的第二个原始矩估计。输入: "s, beta2, t"。输出: "s_corrected"。
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t))

        # 更新参数。输入: "parameters, learning_rate, v_corrected, s_corrected, epsilon"。输出: "parameters"。
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * (v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon))
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * (v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon))

    return parameters, v, s, v_corrected, s_corrected
```
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/621599c7-a181-468d-854b-09d9f6ce5b41)

You now have three working optimization algorithms (mini-batch gradient descent, Momentum, Adam). Let's implement a model with each of these optimizers and observe the difference.

现在您已经有了三种有效的优化算法（迷你批量梯度下降算法、动量算法和亚当算法）。让我们用这三种优化算法分别实现一个模型，并观察它们之间的区别。

<a name='6'></a>  
## 6 - Model with different Optimization algorithms

Below, you'll use the following "moons" dataset to test the different optimization methods. (The dataset is named "moons" because the data from each of the two classes looks a bit like a crescent-shaped moon.) 
```
train_X, train_Y = load_dataset()
```
A 3-layer neural network has already been implemented for you! You'll train it with: 
- Mini-batch **Gradient Descent**: it will call your function:
    - `update_parameters_with_gd()`
- Mini-batch **Momentum**: it will call your functions:
    - `initialize_velocity()` and `update_parameters_with_momentum()`
- Mini-batch **Adam**: it will call your functions:
    - `initialize_adam()` and `update_parameters_with_adam()`

已经为您实现了一个3层神经网络！您将使用以下方法对其进行训练

迷你批量梯度下降：它将调用您的函数：
update_parameters_with_gd()
小批量动量：它将调用您的函数：
initialize_velocity()和update_parameters_with_momentum()
小批量Adam：它将调用您的函数：
initialize_adam()和update_parameters_with_adam()

**这个模型整合了3种optimization**

```python
import matplotlib.pyplot as plt
import numpy as np
import random

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=5000, print_cost=True):
    """
    一个可以在不同优化器模式下运行的三层神经网络模型。
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    optimizer -- the optimizer to be passed, gradient descent, momentum or adam
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(layers_dims)             # 神经网络中的层数
    costs = []                       # 用于记录成本
    t = 0                            # 初始化Adam更新所需的计数器
    seed = 10                        # 为了评估方便，确保你的“随机”小批量与我们的相同
    m = X.shape[1]                   # 训练样本的数量
    
    # Initialize parameters
    parameters = initialize_parameters(layers_dims)  # 初始化参数

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # 梯度下降法不需要额外初始化
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)  # 初始化动量
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)  # 初始化Adam
    
    # Optimization loop
    for i in range(num_epochs):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)  # 随机分割小批量
        cost_total = 0
        
        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch  # 选择一个小批量

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)  # 前向传播

            # Compute cost and add to the cost total
            cost_total += compute_cost(a3, minibatch_Y)  # 计算成本并累加

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)  # 反向传播

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)  # 使用梯度下降更新参数
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)  # 使用动量法更新参数
            elif optimizer == "adam":
                t = t + 1  # Adam计数器
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)  # 使用Adam算法更新参数
        cost_avg = cost_total / m
        
        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))  # 每1000个epoch打印成本值
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
```

Now, run this 3 layer neural network with each of the 3 optimization methods.

<a name='6-1'></a>  
### 6.1 - Mini-Batch Gradient Descent

Run the following code to see how the model does with mini-batch gradient descent.

**首先来看使用常规的Mini-bath gradient descent(准确率70%)**
```python
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```
![6](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/ae8fd125-9f96-4438-a78c-ca698f86ef3f)
![7](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/e0200d0f-a1d0-4650-bec4-90baeec741f8)

<a name='6-2'></a>  
### 6.2 - Mini-Batch Gradient Descent with Momentum

Next, run the following code to see how the model does with momentum. Because this example is relatively simple, the gains from using momemtum are small - but for more complex problems you might see bigger gains.

运行下面的代码查看模型在动量情况下的表现。由于本例相对简单，使用momemtum的收益较小，但对于更复杂的问题，您可能会看到更大的收益。

**在使用Mini-bath和Momentum之后准确率是71%**
```python
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/1f669fc9-4f65-445c-afe8-f81a2c136e96)
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d2f2d2b2-66f6-4bf3-b030-1e8eabcfc6e8)

**最后我再用Mini_bath和Adam结合,准确率最高（94%）**

<a name='6-3'></a>  
### 6.3 - Mini-Batch with Adam

Finally, run the following code to see how the model does with Adam.
```python
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/9da39841-0310-4a98-a538-09fce086c17b)
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a8b1d487-e651-4838-90d2-39e37b6579e2)

<a name='6-4'></a>  
### 6.4 - Summary

<table> 
    <tr>
        <td>
        <b>optimization method</b>
        </td>
        <td>
        <b>accuracy</b>
        </td>
        <td>
        <b>cost shape</b>
        </td>
    </tr>
        <td>
        Gradient descent
        </td>
        <td>
        >71%
        </td>
        <td>
        smooth
        </td>
    <tr>
        <td>
        Momentum
        </td>
        <td>
        >71%
        </td>
        <td>
        smooth
        </td>
    </tr>
    <tr>
        <td>
        Adam
        </td>
        <td>
        >94%
        </td>
        <td>
        smoother
        </td>
    </tr>
</table> 

Momentum usually helps, but given the small learning rate and the simplistic dataset, its impact is almost negligible.

On the other hand, Adam clearly outperforms mini-batch gradient descent and Momentum. If you run the model for more epochs on this simple dataset, all three methods will lead to very good results. However, you've seen that Adam converges a lot faster.

Some advantages of Adam include:

- Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum) 
- Usually works well even with little tuning of hyperparameters (except $\alpha$)


Momentum通常会有所帮助，但由于学习率较低且数据集较为简单，其影响几乎可以忽略不计。

另一方面，Adam明显优于mini-batch梯度下降法和Momentum。如果在这个简单的数据集上运行更多的epochs，这三种方法都会取得非常好的结果。不过，Adam的收敛速度要快得多。

Adam的一些优势包括

- 相对较低的内存需求（尽管高于梯度下降法和带动量的梯度下降法） 
- 即使对超参数进行很少的调整，通常也能很好地工作（$\alpha$除外）

  <a name='7'></a>  
## 7 - Learning Rate Decay and Scheduling

Lastly, the learning rate is another hyperparameter that can help you speed up learning. 

During the first part of training, your model can get away with taking large steps, but over time, using a fixed value for the learning rate alpha can cause your model to get stuck in a wide oscillation that never quite converges. But if you were to slowly reduce your learning rate alpha over time, you could then take smaller, slower steps that bring you closer to the minimum. This is the idea behind learning rate decay. 

Learning rate decay can be achieved by using either adaptive methods or pre-defined learning rate schedules. 

Now, you'll apply scheduled learning rate decay to a 3-layer neural network in three different optimizer modes and see how each one differs, as well as the effect of scheduling at different epochs. 

This model is essentially the same as the one you used before, except in this one you'll be able to include learning rate decay. It includes two new parameters, decay and decay_rate. 


最后，学习率是另一个可以帮助您加快学习速度的超参数。

在训练的最初阶段，您的模型可以采取较大的步长，但随着时间的推移，使用固定值的学习率α会导致您的模型陷入大范围的振荡，永远无法收敛。但是，如果您随着时间的推移慢慢降低学习率α，您就可以迈出更小更慢的步子，从而更接近最小值。这就是学习率衰减背后的理念。

学习率衰减可以通过使用自适应方法或预定义的学习率计划来实现。

现在，您将在三种不同的优化器模式下对一个3层神经网络应用预定学习率衰减，并了解每种模式的不同之处，以及在不同的epochs进行调度的效果。

这个模型与您之前使用的模型基本相同，只是在这个模型中您可以加入学习率衰减。它包括两个新参数，decay和decay_rate。

```python

import matplotlib.pyplot as plt
import numpy as np
import random

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=5000, print_cost=True, decay=None, decay_rate=1):
    """
    3-layer neural network model which can be run in different optimizer modes.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs
    decay -- learning rate decay function, default is None
    decay_rate -- learning rate decay rate, default is 1

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]                   # number of training examples
    lr_rates = []
    learning_rate0 = learning_rate   # the original learning rate
    
    # Initialize parameters
    parameters = initialize_parameters(layers_dims)  # 初始化参数

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)  # 初始化动量
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)  # 初始化Adam
    
    # Optimization loop
    for i in range(num_epochs):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)  # 随机分割小批量
        cost_total = 0
        
        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch  # 选择一个小批量

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)  # 前向传播

            # Compute cost and add to the cost total
            cost_total += compute_cost(a3, minibatch_Y)  # 计算成本并累加

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)  # 反向传播

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)  # 使用梯度下降更新参数
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)  # 使用动量法更新参数
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)  # 使用Adam算法更新参数
        cost_avg = cost_total / m
        if decay:
            learning_rate = decay(learning_rate0, i, decay_rate)  # 学习率衰减
        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))  # 每1000个epoch打印成本值
            if decay:
                print("Learning rate after epoch %i: %f"%(i, learning_rate))  # 若使用学习率衰减，打印学习率
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

```

<a name='7-1'></a>  
### 7.1 - Decay on every iteration  

For this portion of the assignment, you'll try one of the pre-defined schedules for learning rate decay, called exponential learning rate decay. It takes this mathematical form:

$$\alpha = \frac{1}{1 + decayRate \times epochNumber} \alpha_{0}$$

<a name='ex-7'></a>  
### Exercise 7 - update_lr

Epoch number（轮次编号）是指在训练神经网络时，将整个训练数据集（包含多个样本）分成若干个小批量进行反向传播和参数更新的循环次数。在每个轮次（epoch）中，神经网络会遍历整个训练数据集一次。每次遍历一个小批量数据并进行反向传播和参数更新被称为一个迭代（iteration）。当所有小批量数据都被遍历完一次后，完成了一个轮次（epoch）的训练。所以，Epoch number（轮次编号）就是指当前处于第几个训练轮次的编号。通过增加训练轮次的数量，可以提高神经网络的训练效果和性能。

Calculate the new learning rate using exponential weight decay.

# GRADED FUNCTION: update_lr
```python
def update_lr(learning_rate0, epoch_num, decay_rate):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate0 -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer
    decay_rate -- Decay rate. Scalar

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """
    # 计算更新的学习率
    learning_rate = learning_rate0 * np.power(decay_rate, epoch_num)
    
    return learning_rate
```
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/aab5d8b4-9155-44d9-b199-90cbb76d499b)
**带入到上面的3种optimization algorithm中看**
```python
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd", learning_rate = 0.1, num_epochs=5000, decay=update_lr)

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/59aa210e-d20b-43da-acc2-fc9453003e1e)
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/bab22751-20be-40e6-ae9d-69342a9dd54f)

Notice that if you set the decay to occur at every iteration, the learning rate goes to zero too quickly - even if you start with a higher learning rate. 
<table> 
    <tr>
        <td>
        <b>Epoch Number</b>
        </td>
        <td>
        <b>Learning Rate</b>
        </td>
        <td>
        <b>Cost</b>
        </td>
    </tr>
    <tr>
        <td>
        0
        </td>
        <td>
        0.100000
        </td>
        <td>
        0.701091
        </td>
    </tr>
    <tr>
        <td>
        1000
        </td>
        <td>
        0.000100
        </td>
        <td>
        0.661884
        </td>
    </tr>
    <tr>
        <td>
        2000
        </td>
        <td>
        0.000050
        </td>
        <td>
        0.658620
        </td>
    </tr>
    <tr>
        <td>
        3000
        </td>
        <td>
        0.000033
        </td>
        <td>
        0.656765
        </td>
    </tr>
    <tr>
        <td>
        4000
        </td>
        <td>
        0.000025
        </td>
        <td>
        0.655486
        </td>
    </tr>
    <tr>
        <td>
        5000
        </td>
        <td>
        0.000020
        </td>
        <td>
        0.654514
        </td>
    </tr>
</table> 

When you're training for a few epoch this doesn't cause a lot of troubles, but when the number of epochs is large the optimization algorithm will stop updating. One common fix to this issue is to decay the learning rate every few steps. This is called fixed interval scheduling.


<a name='7-2'></a> 
### 7.2 - Fixed Interval Scheduling

You can help prevent the learning rate speeding to zero too quickly by scheduling the exponential learning rate decay at a fixed time interval, for example 1000. You can either number the intervals, or divide the epoch by the time interval, which is the size of window with the constant learning rate. 


您可以在固定时间间隔（例如1000）内安排指数学习率衰减，以防止学习率过快归零。您可以对时间间隔进行编号，也可以将epoch除以时间间隔，即恒定学习率窗口的大小。
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/60be0890-c809-4245-9477-628ca927dbdf)

<a name='ex-8'></a> 
### Exercise 8 - schedule_lr_decay

Calculate the new learning rate using exponential weight decay with fixed interval scheduling.

**Instructions**: Implement the learning rate scheduling such that it only changes when the epochNum is a multiple of the timeInterval.

**Note:** The fraction in the denominator uses the floor operation. 

$$\alpha = \frac{1}{1 + decayRate \times \lfloor\frac{epochNum}{timeInterval}\rfloor} \alpha_{0}$$

**Hint:** [numpy.floor](https://numpy.org/doc/stable/reference/generated/numpy.floor.html)

其中，公式里的[epochNum / timeinterval] 是整除的意思.

# GRADED FUNCTION: schedule_lr_decay
```python

def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate0 -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer.
    decay_rate -- Decay rate. Scalar.
    time_interval -- Number of epochs where you update the learning rate.

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """
    # 计算学习率衰减
    learning_rate = (1 / (1 + decay_rate * (epoch_num // time_interval))) * learning_rate0
    
    return learning_rate
```
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/ab30afa3-0471-4dee-86b7-18aee1b0179f)


这段代码展示了使用 schedule_lr_decay 函数计算学习率衰减的示例，并测试了该函数的输出。

代码的执行过程如下：

首先，将学习率 learning_rate 设置为 0.5，并打印出原始学习率。

然后，定义了两个轮次编号 epoch_num_1 和 epoch_num_2，以及衰减率 decay_rate 和时间间隔 time_interval 的值。

使用 schedule_lr_decay 函数分别计算了经过 epoch_num_1 和 epoch_num_2 轮次后的学习率，将结果分别存储在 learning_rate_1 和 learning_rate_2 变量中。

打印出经过 epoch_num_1 轮次后更新后的学习率和经过 epoch_num_2 轮次后更新后的学习率。

最后，调用 schedule_lr_decay_test 函数来测试 schedule_lr_decay 函数的输出。

这段代码主要用于展示学习率衰减的效果。通过指定不同的轮次编号，可以观察学习率在不同阶段的衰减情况。你可以根据自己的需要修改轮次编号、衰减率和时间间隔的值，并观察学习率的变化。


<a name='7-3'></a> 
### 7.3 - Using Learning Rate Decay for each Optimization Method

Below, you'll use the following "moons" dataset to test the different optimization methods. (The dataset is named "moons" because the data from each of the two classes looks a bit like a crescent-shaped moon.) 

 - 对每种优化方法使用学习率衰减

下面，您将使用下面的 "moons "数据集来测试不同的优化方法。(该数据集之所以命名为 "moons"，是因为来自两个类的数据看起来有点像新月形的月亮）。

<a name='7-3-1'></a> 
#### 7.3.1 - Gradient Descent with Learning Rate Decay
**可以显著的看到准确率提升了，到了94%**

Run the following code to see how the model does gradient descent and weight decay.
```python
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd", learning_rate = 0.1, num_epochs=5000, decay=schedule_lr_decay)

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/232d1a90-ee74-4045-9f07-99348c35cc46)


![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/64f002e1-460e-41eb-9da4-048d3f83ee5a)

<a name='7-3-2'></a> 
#### 7.3.2 - Gradient Descent with Momentum and Learning Rate Decay
**可以看到Momentum的准确率到了95.5%**
Run the following code to see how the model does gradient descent with momentum and weight decay.
```python
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "momentum", learning_rate = 0.1, num_epochs=5000, decay=schedule_lr_decay)

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent with momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/9a5c38c2-74eb-4c42-b70e-2c879e21b6f6)
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/e7b32a2f-fb78-45e0-9659-b293efb1e7aa)

<a name='7-3-3'></a> 
#### 7.3.3 - Adam with Learning Rate Decay

Run the following code to see how the model does Adam and weight decay.
```
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam", learning_rate = 0.01, num_epochs=5000, decay=schedule_lr_decay)

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```
![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/937ff8be-f058-4e07-9dd0-a393e987c90d)
![6](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/247c0543-8bc1-4ec0-9037-35ea585975de)

<a name='7-4'></a> 
### 7.4 - Achieving similar performance with different methods

With Mini-batch GD or Mini-batch GD with Momentum, the accuracy is significantly lower than Adam, but when learning rate decay is added on top, either can achieve performance at a speed and accuracy score that's similar to Adam.

In the case of Adam, notice that the learning curve achieves a similar accuracy but faster.

<table> 
    <tr>
        <td>
        <b>optimization method</b>
        </td>
        <td>
        <b>accuracy</b>
        </td>
    </tr>
        <td>
        Gradient descent
        </td>
        <td>
        >94.6%
        </td>
    <tr>
        <td>
        Momentum
        </td>
        <td>
        >95.6%
        </td>
    </tr>
    <tr>
        <td>
        Adam
        </td>
        <td>
        94%
        </td>
    </tr>
</table> 

使用Mini-batch GD或Mini-batch GD with Momentum，准确率明显低于Adam，但在学习率衰减的基础上，二者都可以达到与Adam相似的速度和准确率。

在Adam的情况下，请注意学习曲线实现了相似的准确率，但速度更快。


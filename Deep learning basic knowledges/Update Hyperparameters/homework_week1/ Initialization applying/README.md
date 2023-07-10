# 使用3种initialization的方法
- *Zeros initialization* --  setting `initialization = "zeros"` in the input argument.
- *Random initialization* -- setting `initialization = "random"` in the input argument. This initializes the weights to large random values.  
- *He initialization* -- setting `initialization = "he"` in the input argument. This initializes the weights to random values scaled according to a paper by He et al., 2015. 

- *Zeros initialization* -- 在输入参数中设置 "initialization = "zeros"。
- 随机初始化* -- 在输入参数中设置 "初始化 = "随机"。这将权重初始化为大的随机值。 
- He initialization* -- 在输入参数中设置 "initialization = "he"。这将权重初始化为根据He等人2015年的一篇论文缩放的随机值。



<a name='1'></a>
## 1 - Packages
```python
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from public_tests import *
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

# load image dataset: blue/red dots in circles
# train_X, train_Y, test_X, test_Y = load_dataset()
```

<a name='2'></a>
## 2 - Loading the Dataset

train_X, train_Y, test_X, test_Y = load_dataset()
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/b3113c82-2e30-442c-a97a-1f1b21e2433f)

You'll use a 3-layer neural network (already implemented for you). These are the initialization methods you'll experiment with: 
- *Zeros initialization* --  setting `initialization = "zeros"` in the input argument.
- *Random initialization* -- setting `initialization = "random"` in the input argument. This initializes the weights to large random values.  
- *He initialization* -- setting `initialization = "he"` in the input argument. This initializes the weights to random values scaled according to a paper by He et al., 2015. 

**Instructions**: Instructions: Read over the code below, and run it. In the next part, you'll implement the three initialization methods that this `model()` calls.

**使用说明**： 说明： 阅读下面的代码，并运行它。在下一部分中，你将实现这个`model()`调用的三个初始化方法。
```python

def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent 
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    Returns:
    parameters -- parameters learnt by the model
    """
        
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]
    
    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)

    for i in range(num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        
        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()    
    return parameters
```
    <a name='4'></a>
## 4 - Zero Initialization

There are two types of parameters to initialize in a neural network:
- the weight matrices $(W^{[1]}, W^{[2]}, W^{[3]}, ..., W^{[L-1]}, W^{[L]})$
- the bias vectors $(b^{[1]}, b^{[2]}, b^{[3]}, ..., b^{[L-1]}, b^{[L]})$

<a name='ex-1'></a>
### Exercise 1 - initialize_parameters_zeros

Implement the following function to initialize all parameters to zeros. You'll see later that this does not work well since it fails to "break symmetry," 
but try it anyway and see what happens. Use `np.zeros((..,..))` with the correct shapes.

练习 1 - 初始化参数为零
执行下面的函数将所有参数初始化为零。稍后你会发现这个函数并不能很好地工作，因为它不能 "打破对称性"，但是还是试试看，看看会发生什么。使用np.zeros((..,...))并使用正确的形状。

**可以从结果清楚的看到，如果全使用np.zero进行归零初始化parameters将使得，每一次训练失去差异性。会导致每次训练的结果都相同**
先执行代码看看结果
# GRADED FUNCTION: initialize_parameters_zeros 
```python
def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims)            # number of layers in the network
    
    for l in range(1, L):
        #(≈ 2 lines of code)
        # parameters['W' + str(l)] = 
        # parameters['b' + str(l)] = 
        # YOUR CODE STARTS HERE
        parameters['W' + str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        # YOUR CODE ENDS HERE
    return parameters
```
```python
parameters = initialize_parameters_zeros([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
initialize_parameters_zeros_test(initialize_parameters_zeros)
```
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/2170475e-41fd-48a3-9e4e-0a72d2a7ae7d)

现在已经得到了包含初始化数据的array，parameters,将它带入到model（）中执行for and back propagation
```python

parameters = model(train_X, train_Y, initialization = "zeros")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/31c06dfb-2133-4a4d-83f8-5d38765a5982)
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/42c8550c-f91b-4399-9733-699a9f2915a4)

**通过cost线图我们可以发现，每次itration产生的cost值都是相同的，这说明，如果我们将weight（W）使用归0法，将无法打破对称性（break symmetry）.**
![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/5d63428a-b0f5-46e1-8055-c07445cbfc87)

__Note__: For sake of simplicity calculations below are done using only one example at a time.

Since the weights and biases are zero, multiplying by the weights creates the zero vector which gives 0 when the activation function is ReLU. As `z = 0`

$$a = ReLU(z) = max(0, z) = 0$$

At the classification layer, where the activation function is sigmoid you then get (for either input): 

$$\sigma(z) = \frac{1}{ 1 + e^{-(z)}} = \frac{1}{2} = y_{pred}$$

As for every example you are getting a 0.5 chance of it being true our cost function becomes helpless in adjusting the weights.

Your loss function:
$$ \mathcal{L}(a, y) =  - y  \ln(y_{pred}) - (1-y)  \ln(1-y_{pred})$$

For `y=1`, `y_pred=0.5` it becomes:

$$ \mathcal{L}(0, 1) =  - (1)  \ln(\frac{1}{2}) = 0.6931471805599453$$

For `y=0`, `y_pred=0.5` it becomes:

$$ \mathcal{L}(0, 0) =  - (1)  \ln(\frac{1}{2}) = 0.6931471805599453$$

As you can see with the prediction being 0.5 whether the actual (`y`) value is 1 or 0 you get the same loss value for both, so none of the weights get adjusted and you are stuck with the same old value of the weights. 

This is why you can see that the model is predicting 0 for every example! No wonder it's doing so badly.

In general, initializing all the weights to zero results in the network failing to break symmetry. This means that every neuron in each layer will learn the same thing, so you might as well be training a neural network with $n^{[l]}=1$ for every layer. This way, the network is no more powerful than a linear classifier like logistic regression. 



<a name='5'></a>
## 5 - Random Initialization

To break symmetry, initialize the weights randomly. Following random initialization, each neuron can then proceed to learn a different function of its inputs. In this exercise, you'll see what happens when the weights are initialized randomly, but to very large values.

<a name='ex-2'></a>
### Exercise 2 - initialize_parameters_random

Implement the following function to initialize your weights to large random values (scaled by \*10) and your biases to zeros. Use `np.random.randn(..,..) * 10` for weights and `np.zeros((.., ..))` for biases. You're using a fixed `np.random.seed(..)` to make sure your "random" weights  match ours, so don't worry if running your code several times always gives you the same initial values for the parameters. 

**5 - 随机初始化**

为了打破对称性，可以随机初始化权重。在随机初始化之后，每个神经元都可以学习其输入的不同函数。在本练习中，您将看到当随机初始化权重，但权重值非常大时，会发生什么情况。
**可以从分类结果清楚的看到，对蓝色的分类效果不好**
**一般来说如果使用了random initialization需要使得这个值尽量小，所以会在随机值定义出来后*0.01**


练习 2 - 随机初始化参数
执行下面的函数，将权重初始化为大的随机值（比例为*10），将偏置初始化为零。权重使用np.random.randn(.,..)*10，偏置使用np.zeros((.,..))。您使用固定的np.random.seed(..)来确保您的 "随机 "权重与我们的相匹配，所以如果多次运行您的代码总是得到相同的参数初始值，也不用担心。


# GRADED FUNCTION: initialize_parameters_random
``` python
def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers
    
    for l in range(1, L):
        #(≈ 2 lines of code)
        # parameters['W' + str(l)] = 
        # parameters['b' + str(l)] =
        # YOUR CODE STARTS HERE
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        # YOUR CODE ENDS HERE

    return parameters
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/347f49fb-47f7-4273-b0b2-7c9346f6a7f3)

**用之前定义的model（）函数对进行了随机初始化的W,b进行iteration**
```python
parameters = model(train_X, train_Y, initialization = "random")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/da649eac-f33d-47c1-a79b-70dbb485a5b3)

**可以看到performance random initialization打破了对称性**

**然后我们使用自定义的边界函数和预测函数来绘制出分类图**

自定义函数setting
**plot_decision_boundary(model, X, y):**

```python

def plot_decision_boundary(model, X, y):
    # 绘制决策边界的代码实现
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision Boundary')
```
**predict_dec(parameters, X):**
```python

def predict_dec(parameters, X):
    # 决策预测的代码实现
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions
```

**绘图：**
```python
plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a0d118f1-14b0-4eb5-a929-eb3a87ec43e2)

**Observations**:
- The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example. Indeed, when $\log(a^{[3]}) = \log(0)$, the loss goes to infinity.
- Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm. 
- If you train this network longer you will see better results, but initializing with overly large random numbers slows down the optimization.

<font color='blue'>
    
**In summary**:
- Initializing weights to very large random values doesn't work well. 
- Initializing with small random values should do better. The important question is, how small should be these random values be? Let's find out up next!

<font color='black'>    
    
**Optional Read:**


The main difference between Gaussian variable (`numpy.random.randn()`) and uniform random variable is the distribution of the generated random numbers:

- numpy.random.rand() produces numbers in a [uniform distribution](https://raw.githubusercontent.com/jahnog/deeplearning-notes/master/Course2/images/rand.jpg).
- and numpy.random.randn() produces numbers in a [normal distribution](https://raw.githubusercontent.com/jahnog/deeplearning-notes/master/Course2/images/randn.jpg).

When used for weight initialization, randn() helps most the weights to Avoid being close to the extremes, allocating most of them in the center of the range.

An intuitive way to see it is, for example, if you take the [sigmoid() activation function](https://raw.githubusercontent.com/jahnog/deeplearning-notes/master/Course2/images/sigmoid.jpg).

You’ll remember that the slope near 0 or near 1 is extremely small, so the weights near those extremes will converge much more slowly to the solution, and having most of them near the center will speed the convergence.


意见：

开始时成本很高。这是因为在随机权值较大的情况下，最后一个激活（sigmoid）输出的结果在某些例子中非常接近0或1，当它弄错例子时，会对该例子产生非常高的损失。事实上，当 ，损失将达到无穷大。
糟糕的初始化会导致梯度消失/爆炸，这也会减慢优化算法的速度。
如果您对该网络进行更长时间的训练，您将看到更好的结果，但使用过大的随机数进行初始化会减慢优化速度。
总之：

将权重初始化为非常大的随机值效果并不好。
使用较小的随机数初始化效果会更好。重要的问题是，这些随机值应该有多小？让我们接下来来了解一下！
可选阅读：

高斯变量（numpy.random.randn()）和均匀随机变量的主要区别在于生成的随机数的分布：

numpy.random.rand()产生的数字是均匀分布的。
和numpy.random.randn()产生正态分布的数字。
当用于权重初始化时，randn()帮助大多数权重避免接近极值，将大多数权重分配在范围的中心。

一个直观的方法是，例如，如果你使用sigmoid()激活函数。

你会记得0或1附近的斜率是非常小的，因此靠近这些极端值的权重会更慢地收敛到解中，而将大部分权重靠近中心会加速收敛。



<a name='6'></a>
## 6 - He Initialization

Finally, try "He Initialization"; this is named for the first author of He et al., 2015. (If you have heard of "Xavier initialization", this is similar except Xavier initialization uses a scaling factor for the weights $W^{[l]}$ of `sqrt(1./layers_dims[l-1])` where He initialization would use `sqrt(2./layers_dims[l-1])`.)

6 - He初始化
最后，试试 "He初始化"；这是以He等人2015年论文的第一作者命名的。(如果您听说过 "Xavier初始化"，那么这与 "He初始化 "类似，只不过Xavier初始化使用了权重的缩放因子，即sqrt(1./layer_dims[l-1])，而 "He初始化 "使用的是sqrt(2./layer_dims[l-1])。

<a name='ex-3'></a>
### Exercise 3 - initialize_parameters_he

Implement the following function to initialize your parameters with He initialization. This function is similar to the previous `initialize_parameters_random(...)`. The only difference is that instead of multiplying `np.random.randn(..,..)` by 10, you will multiply it by $\sqrt{\frac{2}{\text{dimension of the previous layer}}}$, which is what He initialization recommends for layers with a ReLU activation. 

练习 3 - 初始化参数 He
执行下面的函数，用He初始化来初始化参数。这个函数类似于前面的 initialize_parameters_random(...)。唯一不同的是，不是将np.random.randn(..,..)乘以10，而是乘以上一层的2dimension， $\sqrt{\frac{2}{\text{dimension of the previous layer}}}$， 这也是He初始化对ReLU激活层的建议。

**iteration到底多少次能找到合适的parameters，到底怎么才算合适？**
一般只有在debug或者是需要证明准确性的时候，才会进行.
Implement gradient checking to verify the accuracy of your backprop implementation.

我们使用Numerical Approximation of gradient这个方法来确定是否合适.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/f00ae361-e6cd-4e39-86e2-3b137d593dbc)


<a name='1'></a>
## 1 - Packages
```python
import numpy as np
from testCases import *
from public_tests import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

%load_ext autoreload
%autoreload 2
```

<a name='2'></a>
## 2 - Problem Statement

You are part of a team working to make mobile payments available globally, and are asked to build a deep learning model to detect 
fraud--whenever someone makes a payment, you want to see if the payment might be fraudulent, such as if the user's account has been taken over by a hacker.

You already know that backpropagation is quite challenging to implement, and sometimes has bugs. Because this is a mission-critical application, 
your company's CEO wants to be really certain that your implementation of backpropagation is correct. Your CEO says,
"Give me proof that your backpropagation is actually working!" To give this reassurance, you are going to use "gradient checking."

您是一个致力于在全球范围内实现移动支付的团队的一员，该团队要求您建立一个深度学习模型来检测欺诈行为。
欺诈--当有人进行支付时，您想知道该支付是否可能是欺诈性的，比如用户的账户是否被黑客盗用。

您已经知道反向传播的实现具有相当的挑战性，有时还会出现错误。因为这是一个关键任务应用程序、 
因为这是一个关键任务应用，贵公司的首席执行官希望真正确定您的反向传播实现是正确的。您的首席执行官说
"给我证明你们的反向传播确实有效！" 为了保证这一点，你将使用 "梯度检查"。

<a name='3'></a>
## 3 - How does Gradient Checking work?
Backpropagation computes the gradients $\frac{\partial J}{\partial \theta}$, where $\theta$ denotes the parameters of the model. $J$ is computed using forward propagation and your loss function.

Because forward propagation is relatively easy to implement, you're confident you got that right, and so you're almost 100% sure that you're computing the cost $J$ correctly. Thus, you can use your code for computing $J$ to verify the code for computing $\frac{\partial J}{\partial \theta}$.

Let's look back at the definition of a derivative (or gradient):$$ \frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} \tag{1}$$

If you're not familiar with the "$\displaystyle \lim_{\varepsilon \to 0}$" notation, it's just a way of saying "when $\varepsilon$ is really, really small."

You know the following:

$\frac{\partial J}{\partial \theta}$ is what you want to make sure you're computing correctly.
You can compute $J(\theta + \varepsilon)$ and $J(\theta - \varepsilon)$ (in the case that $\theta$ is a real number), since you're confident your implementation for $J$ is correct.
Let's use equation (1) and a small value for $\varepsilon$ to convince your CEO that your code for computing $\frac{\partial J}{\partial \theta}$ is correct!

3 - 梯度检查是如何工作的？
反向传播计算梯度 ∂𝐽∂𝜃 ，其中 𝜃 表示模型参数。 𝐽使用前向传播和损失函数计算。

由于前向传播比较容易实现，您确信您的计算是正确的，因此您几乎可以 100% 地确定您计算的代价𝐽 是正确的。因此，您可以使用计算𝐽 的代码来验证计算∂𝐽∂𝜃 的代码。

让我们回顾一下导数（或梯度）的定义：
∂𝐽∂𝜃=lim𝜀→0𝐽(𝜃+𝜀)-𝐽(𝜃-𝜀)2𝜀(1)
如果您不熟悉 "lim𝜀→0 "这个符号，它只是 "当𝜀 非常非常小的时候 "的一种说法。

您知道以下内容：

∂𝐽∂𝜃 是您需要确保计算正确的值。您可以计算𝐽(𝜃+𝜀)和𝐽(𝜃-𝜀)（在𝜃为实数的情况下），因为您确信您对𝐽的实现是正确的。让我们使用等式 (1) 和一个较小的 𝜀 值来使您的首席执行官相信您计算 ∂𝐽∂𝜃 的代码是正确的！


<a name='4'></a>
## 4 - 1-Dimensional Gradient Checking

Consider a 1D linear function $J(\theta) = \theta x$. The model contains only a single real-valued parameter $\theta$, and takes $x$ as input.

You will implement code to compute $J(.)$ and its derivative $\frac{\partial J}{\partial \theta}$. You will then use gradient checking to make sure your derivative computation for $J$ is correct. 

![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/7712d147-86cf-4c73-9748-fd74f2706b78)

<caption><center><font color='purple'><b>Figure 1</b>:1D linear model </font></center></caption>

The diagram above shows the key computation steps: First start with $x$, then evaluate the function $J(x)$ ("forward propagation"). Then compute the derivative $\frac{\partial J}{\partial \theta}$ ("backward propagation"). 

<a name='ex-1'></a>
### Exercise 1 - forward_propagation

Implement `forward propagation`. For this simple function compute $J(.)$


**4 - 一维梯度检验**

考虑一维线性函数 𝐽(𝜃)=𝜃𝑥 。该模型只包含一个实值参数 𝜃 ，并将 𝑥 作为输入。

您将执行代码来计算 𝐽(.) 及其导数 ∂𝐽∂𝜃 。然后您将使用梯度检查来确保您对 𝐽 的导数计算是正确的。

上图显示了关键的计算步骤： 首先从 𝑥 开始，然后评估函数 𝐽(𝑥)（"前向传播"）。然后计算导数 ∂𝐽∂𝜃（"反向传播"）。


练习 1 - 正向传播
实现正向传播。对于这个简单函数，计算 𝐽(.)

# GRADED FUNCTION: forward_propagation
```python
def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    J -- the value of function J, computed using the formula J(theta) = theta * x
    """
    
    # (approx. 1 line)
    # J = 
    # YOUR CODE STARTS HERE
    
    J = theta * x

    # YOUR CODE ENDS HERE
    
    return J
```
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/32226034-025f-4816-8ab1-a5f330e8440f)


<a name='ex-2'></a>
### Exercise 2 - backward_propagation

Now, implement the `backward propagation` step (derivative computation) of Figure 1.
That is, compute the derivative of $J(\theta) = \theta x$ with respect to $\theta$. To save you from doing the calculus, you should get $dtheta = \frac { \partial J }{ \partial \theta} = x$.

<a name='ex-2'></a>
###练习 2 - 向后传播

现在，实现图1的 "反向传播 "步骤（导数计算）。也就是说，计算$J(\theta) = \theta x$相对于$\theta$的导数。为了节省您的微积分计算，
您应该得到 $dtheta = \frac { \partial J }{ \partial \theta} = x$。

# GRADED FUNCTION: backward_propagation
```python
def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta (see Figure 1).
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    dtheta -- the gradient of the cost with respect to theta
    """
    
    # (approx. 1 line)
    # dtheta = 
    # YOUR CODE STARTS HERE
    dtheta = x
    
    # YOUR CODE ENDS HERE
    
    return dtheta
```
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/2d6babe8-9960-40c1-8576-6d67482ca304)


<a name='ex-3'></a>
### Exercise 3 - gradient_check

To show that the `backward_propagation()` function is correctly computing the gradient $\frac{\partial J}{\partial \theta}$, let's implement gradient checking.

**Instructions**:
- First compute "gradapprox" using the formula above (1) and a small value of $\varepsilon$. Here are the Steps to follow:
    1. $\theta^{+} = \theta + \varepsilon$
    2. $\theta^{-} = \theta - \varepsilon$
    3. $J^{+} = J(\theta^{+})$
    4. $J^{-} = J(\theta^{-})$
    5. $gradapprox = \frac{J^{+} - J^{-}}{2  \varepsilon}$
- Then compute the gradient using backward propagation, and store the result in a variable "grad"
- Finally, compute the relative difference between "gradapprox" and the "grad" using the following formula:
$$ difference = \frac {\mid\mid grad - gradapprox \mid\mid_2}{\mid\mid grad \mid\mid_2 + \mid\mid gradapprox \mid\mid_2} \tag{2}$$
You will need 3 Steps to compute this formula:
   - 1'. compute the numerator using np.linalg.norm(...)
   - 2'. compute the denominator. You will need to call np.linalg.norm(...) twice.
   - 3'. divide them.
- If this difference is small (say less than $10^{-7}$), you can be quite confident that you have computed your gradient correctly. Otherwise, there may be a mistake in the gradient computation. 

# GRADED FUNCTION: gradient_check
```python
def gradient_check(x, theta, epsilon=1e-7, print_msg=False):
    """
    Implement the gradient checking presented in Figure 1.
    
    Arguments:
    x -- a float input
    theta -- our parameter, a float as well
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient. Float output
    """
    
    # Compute gradapprox using right side of formula (1). epsilon is small enough, you don't need to worry about the limit.
    # (approx. 5 lines)
    # theta_plus =                                 # Step 1
    # theta_minus =                                # Step 2
    # J_plus =                                    # Step 3
    # J_minus =                                   # Step 4
    # gradapprox =                                # Step 5
    # YOUR CODE STARTS HERE
    theta_plus = theta + epsilon
    theta_minus = theta - epsilon
    J_plus = forward_propagation(x, theta_plus)
    J_minus = forward_propagation(x, theta_minus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)
    
    # YOUR CODE ENDS HERE
    
    # Check if gradapprox is close enough to the output of backward_propagation()
    #(approx. 1 line) DO NOT USE "grad = gradapprox"
    # grad =
    # YOUR CODE STARTS HERE
    
    grad = backward_propagation(x, theta)
    # YOUR CODE ENDS HERE
    
    #(approx. 3 lines)
    # numerator =                                 # Step 1'
    # denominator =                               # Step 2'
    # difference =                                # Step 3'
    # YOUR CODE STARTS HERE
    numerator = np.linalg.norm(gradapprox - dtheta)
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(dtheta)
    difference = numerator / denominator
    
    # YOUR CODE ENDS HERE
    if print_msg:
        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference
```
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c4b8b67a-bfa8-439d-aceb-9b98e5893784)
**结果偏差小于10的-7，好结果**
Congrats, the difference is smaller than the $2 * 10^{-7}$ threshold. So you can have high confidence that you've correctly computed the gradient in `backward_propagation()`. 

Now, in the more general case, your cost function $J$ has more than a single 1D input. When you are training a neural network, $\theta$ actually consists of multiple matrices
$W^{[l]}$ and biases $b^{[l]}$! It is important to know how to do a gradient check with higher-dimensional inputs. Let's do it!


恭喜你，差值小于2 * 10^{-7}$阈值。所以你可以很有信心地认为你在`backward_propagation()`中正确地计算了梯度。

现在，在更一般的情况下，您的代价函数$J$有不止一个1D输入。在训练神经网络时，$theta$实际上由多个矩阵组成
W^{[l]}$和偏置$b^{[l]}$！了解如何对高维输入进行梯度检查非常重要。让我们开始吧！

<a name='5'></a>
## 5 - N-Dimensional Gradient Checking
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/98d93ec2-95ba-4405-a4a6-963200bd0fa9)

Let's look at your implementations for forward propagation and backward propagation. 

```python
def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.
    
    Arguments:
    X -- training set for m examples
    Y -- labels for m examples 
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    
    Returns:
    cost -- the cost function (logistic cost for m examples)
    cache -- a tuple with the intermediate values (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    """
    
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    log_probs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(log_probs)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return cost, cache
```
Now, run backward propagation.
```
def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    cache -- cache output from forward_propagation_n()
    
    Returns:
    gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) * 2
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```

您在欺诈检测测试集上获得了一些结果，但您并不100%确定您的模型。人无完人！让我们实施梯度检查来验证您的梯度是否正确。

**How does gradient checking work?**.

As in Section 3 and 4, you want to compare "gradapprox" to the gradient computed by backpropagation. The formula is still:

$$ \frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} \tag{1}$$

However, $\theta$ is not a scalar anymore. It is a dictionary called "parameters". The  function "`dictionary_to_vector()`" has been implemented for you. It converts the "parameters" dictionary into a vector called "values", obtained by reshaping all parameters (W1, b1, W2, b2, W3, b3) into vectors and concatenating them.

The inverse function is "`vector_to_dictionary`" which outputs back the "parameters" dictionary.

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/e40ba432-e9aa-49b1-a228-c694637c978f)

gradients_to_vector() 已经将 "gradients "字典转换为矢量 "grad"，因此您不必担心。

现在，对于矢量中的每一个参数，您都将应用与 gradient_check 练习相同的步骤。如果检查结果符合预期，近似值中的每个值都必须与 grad 向量中存储的实际梯度值相匹配。

请注意，grad 是使用函数 gradients_too_vector 计算的，该函数使用了 backward_propagation_n 函数的梯度输出。

<a name='ex-4'></a>
### Exercise 4 - gradient_check_n

Implement the function below.

**Instructions**: Here is pseudo-code that will help you implement the gradient check.

For each i in num_parameters:
- To compute `J_plus[i]`:
    1. Set $\theta^{+}$ to `np.copy(parameters_values)`
    2. Set $\theta^{+}_i$ to $\theta^{+}_i + \varepsilon$
    3. Calculate $J^{+}_i$ using to `forward_propagation_n(x, y, vector_to_dictionary(`$\theta^{+}$ `))`.     
- To compute `J_minus[i]`: do the same thing with $\theta^{-}$
- Compute $gradapprox[i] = \frac{J^{+}_i - J^{-}_i}{2 \varepsilon}$
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/93fdbdf3-5ff6-48e7-a717-a0f4f6b544af)


**Note**: Use `np.linalg.norm` to get the norms


# GRADED FUNCTION: gradient_check_n
```python
def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7, print_msg=False):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters 
    X -- input datapoint, of shape (input size, number of examples)
    Y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have outputs two parameters but we only care about the first one
        #(approx. 3 lines)
        # theta_plus =                                        # Step 1
        # theta_plus[i] =                                     # Step 2
        # J_plus[i], _ =                                     # Step 3
        # YOUR CODE STARTS HERE
        theta_plus = np.copy(parameters_values)
        theta_plus[i][0] += epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))
        
        # YOUR CODE ENDS HERE
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        #(approx. 3 lines)
        # theta_minus =                                    # Step 1
        # theta_minus[i] =                                 # Step 2        
        # J_minus[i], _ =                                 # Step 3
        # YOUR CODE STARTS HERE
        theta_minus = np.copy(parameters_values)
        theta_minus[i][0] -= epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))
        
        # YOUR CODE ENDS HERE
        
        # Compute gradapprox[i]
        # (approx. 1 line)
        # gradapprox[i] = 
        # YOUR CODE STARTS HERE
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

        # YOUR CODE ENDS HERE
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    # (approx. 3 line)
    # numerator =                                             # Step 1'
    # denominator =                                           # Step 2'
    # difference =                                            # Step 3'
    # YOUR CODE STARTS HERE
        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
        difference = numerator / denominator
            
    
    # YOUR CODE ENDS HERE
    if print_msg:
        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference
```
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/8dcf8d42-d76b-4081-8178-4e9773ba81df)


It seems that there were errors in the `backward_propagation_n` code! Good thing you've implemented the gradient check. Go back to `backward_propagation` and try to find/correct the errors *(Hint: check dW2 and db1)*. Rerun the gradient check when you think you've fixed it. Remember, you'll need to re-execute the cell defining `backward_propagation_n()` if you modify the code. 

Can you get gradient check to declare your derivative computation correct? Even though this part of the assignment isn't graded, you should try to find the bug and re-run gradient check until you're convinced backprop is now correctly implemented. 

**Notes** 
- Gradient Checking is slow! Approximating the gradient with $\frac{\partial J}{\partial \theta} \approx  \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon}$ is computationally costly. For this reason, we don't run gradient checking at every iteration during training. Just a few times to check if the gradient is correct. 
- Gradient Checking, at least as we've presented it, doesn't work with dropout. You would usually run the gradient check algorithm without dropout to make sure your backprop is correct, then add dropout. 

Congrats! Now you can be confident that your deep learning model for fraud detection is working correctly! You can even use this to convince your CEO. :) 
<br>
<font color='blue'>
    
**What you should remember from this notebook**:
- Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
- Gradient checking is slow, so you don't want to run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.


似乎在backward_propagation_n代码中出现了错误！好在你已经实现了梯度检查。返回到 backward_propagation，尝试查找/纠正错误（提示：检查 dW2 和 db1）。当你认为已经解决了问题时，重新运行梯度检查。记住，如果您修改了代码，您需要重新执行定义 backward_propagation_n() 的单元格。

你能让梯度检查声明你的导数计算是正确的吗？尽管这部分作业没有评分，您还是应该尝试找到错误并重新运行梯度检查，直到您确信 backprop 已经正确实现。

注意事项

梯度检查很慢！用∂𝐽∂𝜃≈𝐽(𝜃+𝜀)-𝐽(𝜃-𝜀)2𝜀近似梯度的计算代价很高。因此，我们在训练过程中不会每次迭代都进行梯度检查。只需要检查几次梯度是否正确。
梯度检查，至少我们所介绍的梯度检查，并不适用于 dropout。通常情况下，您需要在不使用dropout的情况下运行梯度检查算法，以确保您的backprop是正确的，然后再添加dropout。
恭喜您 现在您可以确信您的欺诈检测深度学习模型工作正常了！您甚至可以用它来说服您的首席执行官。）

您应该记住本笔记本中的内容：

梯度检查验证反向传播的梯度与梯度的数值近似值（使用正向传播计算）之间的接近程度。
梯度检查的速度较慢，因此您不希望在每次训练迭代中都运行梯度检查。You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.

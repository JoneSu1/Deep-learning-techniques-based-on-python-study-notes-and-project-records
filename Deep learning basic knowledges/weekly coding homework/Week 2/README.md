####介绍构建sigmoid函数的方法（numpy）

如果只是需要对单个数字，而不是数组进行sigmoid求值，可以使用python内置的library： math.exp（）来进行

      import math
      """
      Compute sigmoid of x.

      Arguments:
      x -- A scalar

      Return:
      s -- sigmoid(x)
      """
      def basic_sigmoid(x):

        s = 1 / (1 + math.exp(-x))
        return s

      print("basic_sigmoid(1) =", basic_sigmoid(1))

如果是相对数列（verctor）进行求sigmoid， 就需要用到library： numpy中的，np.exp()


        import numpy as np

        # example of np.exp
        t_x = np.array([1, 2, 3])
        print(np.exp(t_x)) # result is (exp(1), exp(2), exp(3))

        output
        
        [ 2.71828183  7.3890561  20.08553692]

----------------------------------------------------------

###降维 Sigmoid Gradient
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/5c749a00-16a8-4504-bcbf-887ed1652db5)


      当计算出激活函数sigmoid的sloop或者说derivative之后就可以根据这个来对参数进行调整并优化模型.

CODE

            # GRADED FUNCTION: sigmoid_derivative

            def sigmoid_derivative(x):
                """
                Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
                You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
                Arguments:
                x -- A scalar or numpy array

                Return:
                ds -- Your computed gradient.
                """
    
                #(≈ 2 lines of code)
                # s = 
                # ds = 
                # YOUR CODE STARTS HERE
                s = 1 / (1 + np.exp(-x))
                ds = s * (1 - s)
    
                # YOUR CODE ENDS HERE
    
                return ds

                t_x = np.array([1, 2, 3])
                print ("sigmoid_derivative(t_x) = " + str(sigmoid_derivative(t_x)))

                sigmoid_derivative_test(sigmoid_derivative)
                  Output
                  sigmoid_derivative(t_x) = [0.19661193 0.10499359 0.04517666]

###### np.reshape 在深度学习中的重要性

![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/edb7a24d-1713-41a7-b243-8c3e8da1ee76)

![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/4bc643f4-17de-4ad9-905e-a0e6222fe71f)

**Notice**

在转换3维array的时候，v = v.reshape((v.shape[0] * v.shape[1], v.shape[2])) 在括号最后要加上1，表示转换成一个列向量.

CODE

             # GRADED FUNCTION:image2vector

             def image2vector(image):
                 """
                 Argument:
                 image -- a numpy array of shape (length, height, depth)
    
                 Returns:
                 v -- a vector of shape (length*height*depth, 1)
                 """
    
                 # (≈ 1 line of code)
                 # v =
                 # YOUR CODE STARTS HERE
                 v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    
                 # YOUR CODE ENDS HERE
    
                 return v
                 # This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
                 t_image = np.array([[[ 0.67826139,  0.29380381],
                                      [ 0.90714982,  0.52835647],
                                      [ 0.4215251 ,  0.45017551]],

                                    [[ 0.92814219,  0.96677647],
                                     [ 0.85304703,  0.52351845],
                                     [ 0.19981397,  0.27417313]],
                                    [[ 0.60659855,  0.00533165],
                                     [ 0.10820313,  0.49978937],
                                     [ 0.34144279,  0.94630077]]])

                 print ("image2vector(image) = " + str(image2vector(t_image)))

                 image2vector_test(image2vector)

                 Output
                 image2vector(image) = [[0.67826139]
                  [0.29380381]
                  [0.90714982]
                  [0.52835647]
                  [0.4215251 ]
                  [0.45017551]
                  [0.92814219]
                  [0.96677647]
                  [0.85304703]
                  [0.52351845]
                  [0.19981397]
                  [0.27417313]
                  [0.60659855]
                  [0.00533165]
                  [0.10820313]
                  [0.49978937]
                  [0.34144279]
                  [0.94630077]]
                  All tests passed.

#### 由于我们有时处理的数据量太大，在进行gradient 之前进行normolization可以加快.

由于我们会把数据都归到一列或者一行，所以在进行归一化的时候.
是先对array中的数，进行取平方和再开根。 使用的是np中的x_nor = np.linalg.norm(x, ord=2, axis=1, keepdims=True).
其中ord=2表示取平方和，再开根号.
axis=1表示，计算行。如果axis=0则表示计算列.
keepdims=True: 指定是否保持结果数组的维度。设置为 True 时，结果数组会保持原始输入的维度，其中被归约的维度会被保留为长度为 1 的维度。

当得到了x_nor这一个标准值之后. 我们用向量x除以x_nor来保证x向量中的每一个元素都是标准值的.

Code

            # GRADED FUNCTION: normalize_rows

            def normalize_rows(x):
                """
                Implement a function that normalizes each row of the matrix x (to have unit length).
    
                Argument:
                x -- A numpy matrix of shape (n, m)
    
                Returns:
                x -- The normalized (by row) numpy matrix. You are allowed to modify x.
                """
    
                #(≈ 2 lines of code)
                # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
                # x_norm =
                # Divide x by its norm.
                # x =
                # YOUR CODE STARTS HERE
                x_norm = np.linalg.norm(x, ord= 2, axis = 1, keepdims = True)
                x = x/x_norm
    
                # YOUR CODE ENDS HERE

                return x

                x = np.array([[0, 3, 4],
                          [1, 6, 4]])
            print("normalizeRows(x) = " + str(normalize_rows(x)))

            normalizeRows_test(normalize_rows)

            Output
            normalizeRows(x) = [[0.         0.6        0.8       ]
             [0.13736056 0.82416338 0.54944226]]
             All tests passed.

### 谈论多分类问题的激活函数，softmax，刚刚讨论的是用于二分类的sigmoid函数.
.
Softmax 函数的输出范围也是 (0, 1)，但它是将一组实数转换为概率分布，输出的每个元素表示对应类别的概率，并且所有元素的和等于 1。它常用于多类别分类问题。

Sigmoid 函数在输入很大或很小的情况下，梯度会趋近于零，称为梯度消失问题。这可能导致网络训练过程中的梯度不稳定性。
Softmax 函数的梯度计算相对稳定，没有明显的梯度消失问题。
总的来说，Sigmoid 函数适用于二分类问题，而 Softmax 函数适用于多类别分类问题。Softmax 函数提供了更明确的类别概率信息，适用于多类别分类任务。
在深度学习中，Softmax 函数通常用于最后一层的输出层，而 Sigmoid 函数可以在中间层或输出层中使用。

**它特别像是多个按列排的array组合了起来，而softmax函数会将每一列看作一组，然后分别进行分类判断，分别进行归一化**
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/4d0a9250-8477-497c-b446-8d7c2c01cb58)

 其中Softmax函数是直接以e为底的指数的形式，放大数据，将优势的数据放大，劣势的缩小。
 Softmax 函数的定义是：softmax(x) = exp(x) / sum(exp(x))。
在 softmax 函数中，我们使用指数函数来进行归一化操作。对于每个输入元素，我们将其应用指数函数，以增强大的元素并抑制小的元素。
然后，我们将所有指数化值的总和作为分母，将每个指数化值除以总和以获得每个输入值对应的概率分布.

、因为softmax是输出了一个概率分布，所以softmax的结果相加是等于1的.


What you need to remember:
np.exp(x) works for any np.array x and applies the exponential function to every coordinate
the sigmoid function and its gradient
image2vector is commonly used in deep learning
np.reshape is widely used. In the future, you'll see that keeping your matrix/vector dimensions straight will go toward eliminating a lot of bugs.
numpy has efficient built-in functions
broadcasting is extremely useful

Coding

            # GRADED FUNCTION: softmax

            def softmax(x):
                """Calculates the softmax for each row of the input x.

                Your code should work for a row vector and also for matrices of shape (m,n).
            
                Argument:
                x -- A numpy matrix of shape (m,n)

                Returns:
            s -- A numpy matrix equal to the softmax of x, of shape (m,n)
                """
    
                #(≈ 3 lines of code)
                # Apply exp() element-wise to x. Use np.exp(...).
                # x_exp = ...

                # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
                # x_sum = ...
    
                # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
                # s = ...
    
                # YOUR CODE STARTS HERE
                x_exp = np.exp(x)
    
                x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    
                s = x_exp / x_sum
    
                # YOUR CODE ENDS HERE
    
                return s

                t_x = np.array([[9, 2, 5, 0, 0],
                [7, 5, 0, 0 ,0]])
            print("softmax(x) = " + str(softmax(t_x)))

            softmax_test(softmax)

            Output

            softmax(x) = [[9.80897665e-01 8.94462891e-04 1.79657674e-02 1.21052389e-04
              1.21052389e-04]
             [8.78679856e-01 1.18916387e-01 8.01252314e-04 8.01252314e-04
              8.01252314e-04]]
             All tests passed.

### verctorization

在深度学习中，你要处理非常大的数据集。因此，非计算最优函数可能成为算法中的巨大瓶颈，并可能导致模型需要很长时间才能运行。
为了确保代码的计算效率，您将使用向量化。例如，试着分辨以下点/外/元素积的实现之间的区别。


### 关于L1的loss值再numpy中的实现. （L1绝对值损失）

实现L1损耗的numpy矢量化版本。你可能会发现函数abs(x) (x的绝对值)很有用，提醒一下，损失是用来评估模型的性能的。你的损失越大，你的预测(y)与真实值(y)的差异就越大。在深度学习中，你使用梯度下降(Gradient Descent)等优化算法来训练你的模型并最小化成本。L1损耗定义为:
![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/2f42edf6-6f13-4f76-bdd2-77c1439e69c4)

所以再numpy中构建函数就需要用到abs（）绝对值函数， np.sum() 计算array元素之和的函数.

Coding

             # GRADED FUNCTION: L1

             def L1(yhat, y):
                 """
                 Arguments:
                 yhat -- vector of size m (predicted labels)
                 y -- vector of size m (true labels)
    
                 Returns:
                 loss -- the value of the L1 loss function defined above
                 """
    
                 #(≈ 1 line of code)
                 # loss = 
                 # YOUR CODE STARTS HERE
                 loss = np.sum(np.abs(y - yhat))
    
                 # YOUR CODE ENDS HERE
                 
                 return loss

                 yhat = np.array([.9, 0.2, 0.1, .4, .9])
             y = np.array([1, 0, 0, 1, 1])
             print("L1 = " + str(L1(yhat, y)))

             L1_test(L1)

             output
             L1 = 1.1
              All tests passed.

### L2 loss值的计算，L2（平方损失）

实现L2损耗的numpy矢量化版本。有几种实现L2损失的方法，但您可能会发现函数np.dot()很有用。
提醒一下.![6](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/788fcc09-9530-4d20-b0a6-0d7be8ee4155)

**由于是平方损失公式，所以我们需要用到np.square()函数来是input达到平方的效果，也需要使用np.sum（）达到累加的效果**

Coding
             # GRADED FUNCTION: L2

             def L2(yhat, y):
                 """
                 Arguments:
                 yhat -- vector of size m (predicted labels)
                 y -- vector of size m (true labels)
    
                 Returns:
                 loss -- the value of the L2 loss function defined above
                 """
    
                 #(≈ 1 line of code)
                 # loss = ...
                 # YOUR CODE STARTS HERE
    
                 loss = np.sum(np.square(y - yhat))
                 # YOUR CODE ENDS HERE
    
                 return loss

                 yhat = np.array([.9, 0.2, 0.1, .4, .9])
             y = np.array([1, 0, 0, 1, 1])

             print("L2 = " + str(L2(yhat, y)))

             L2_test(L2)

             Output
             L2 = 0.43
              All tests passed.

使用时候的提醒：
选择使用 L1 Loss 还是 L2 Loss 取决于具体的应用场景和任务要求。


L1 Loss 适用于以下情况：

稀疏性要求：当希望模型的结果具有稀疏性时，即希望大部分权重为零，可以使用 L1 Loss。因为 L1 Loss 对于较小的权重值施加较大的惩罚，会促使模型学习到更多的零权重。

特征选择：L1 Loss 可以用作特征选择的一种方法。通过最小化 L1 Loss，可以将与目标变量无关的特征的权重调整为零，从而实现特征选择的效果。

鲁棒性：L1 Loss 对于异常值（离群点）具有较好的鲁棒性。由于 L1 Loss 是绝对值之和，对于较大的异常值，其误差项对总损失的贡献较大，从而使模型更加关注这些异常值。

L2 Loss 适用于以下情况：

平滑性要求：当希望模型的结果具有平滑性时，即希望权重分布相对均匀，可以使用 L2 Loss。因为 L2 Loss 对较大的权重施加较大的惩罚，可以减小权重的波动。

回归问题：在回归问题中，L2 Loss 常用于衡量预测值与真实值之间的差异。通过最小化 L2 Loss，可以使预测值更接近真实值。

模型复杂度控制：L2 Loss 在正则化（regularization）中起到控制模型复杂度的作用。通过在损失函数中添加 L2 Loss 项，可以限制模型权重的大小，避免过拟合问题。

需要根据具体问题的要求和特点选择适合的损失函数。在某些情况下，也可以结合使用 L1 Loss 和 L2 Loss，形成 L1+L2 Loss，以综合考虑稀疏性和平滑性的需求。

--------------------------------------------------
### 实践使用神经网络的逻辑回归构建一个对猫进行分类的分类器.

I will build a logical regression classifiler to recognize cats.

You will learn to:

Build the general architecture of a learning algorithm, including:
Initializing parameters
Calculating the cost function and its gradient
Using an optimization algorithm (gradient descent)
Gather all three functions above into a main model function, in the right order.

<a name='1'></a>
## 1 - Packages ##

First, let's run the cell below to import all the packages that you will need during this assignment. 
- [numpy](https://numpy.org/doc/1.20/) is the fundamental package for scientific computing with Python.
- [h5py](http://www.h5py.org) is a common package to interact with a dataset that is stored on an H5 file.
- [matplotlib](http://matplotlib.org) is a famous library to plot graphs in Python.
- [PIL](https://pillow.readthedocs.io/en/stable/) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end.

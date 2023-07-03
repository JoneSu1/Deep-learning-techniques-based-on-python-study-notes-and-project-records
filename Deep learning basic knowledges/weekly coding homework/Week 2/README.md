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

-NumPy：NumPy 是科学计算的基础包，提供了强大的数组操作功能。在神经网络中，NumPy 用于处理和操作多维数组，执行向量化计算，
以及进行矩阵运算和数学函数的计算。它为神经网络的数据处理和数值计算提供了高效的工具。

-h5py：h5py 是一个常用的与存储在 H5 文件中的数据集进行交互的库。在神经网络中，可以使用 h5py 读取和写入包含大量数据的数据集，
以便进行模型的训练和评估。它提供了对 H5 文件的高级接口，使得数据的读取和写入更加方便和高效。

-Matplotlib：Matplotlib 是一个著名的绘图库，用于在 Python 中绘制各种类型的图形和可视化。在神经网络中，
Matplotlib 可以用于绘制损失函数的曲线图、展示模型的性能指标、可视化数据集等。它提供了丰富的绘图功能，可以帮助我们更好地理解和展示神经网络的结果和过程。

-PIL（Python Imaging Library）和 SciPy：PIL 是一个图像处理库，而 SciPy 是一个科学计算库。在神经网络中，PIL 和 SciPy 可以用于处理图像数据，
进行图像的加载、预处理、转换等操作。例如，可以使用 PIL 将图像转换为适用于神经网络的数组表示，或者使用 SciPy 进行图像的滤波、缩放、旋转等处理操作。

            import numpy as np
            import copy
            import matplotlib.pyplot as plt
            import h5py
            import scipy
            from PIL import Image
            from scipy import ndimage
            from lr_utils import load_dataset
            from public_tests import *

            %matplotlib inline
            %load_ext autoreload
            %autoreload 2

<a name='2'></a>
## 2 - Overview of the Problem set ##

**Problem Statement**: You are given a dataset ("data.h5") containing:
    - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
    - a test set of m_test images labeled as cat or non-cat
    - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).

You will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.

Let's get more familiar with the dataset. Load the data by running the following code.

       # Loading the data (cat/non-cat)
       train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

我们在图像数据集(训练和测试)的末尾添加了“_origin”，因为我们要对它们进行预处理。预处理之后，
我们将得到train_set_x和test_set_x(标签train_set_y和test_set_y不需要任何预处理)。train_set_x_origin和test_set_x_origin
的每一行都是一个表示图像的数组。您可以通过运行以下代码来可视化示例。您也可以随意更改索引值并重新运行以查看其他图像。

代码解释：
- index = 25：选择要显示的图像的索引号，这里选择了索引为25的图像。
- plt.imshow(train_set_x_orig[index])：使用Matplotlib的imshow函数显示指定索引的图像。train_set_x_orig是一个包含训练图像数据的NumPy数组，
 通过使用索引号index可以获取到对应的图像数据。
- print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")：
 打印出图像对应的标签信息。train_set_y是一个包含训练集标签的NumPy数组，通过使用索引号index可以获取到对应的标签数据。classes是一个包含类别名称的列表或数组，
通过将标签索引号np.squeeze(train_set_y[:, index])作为索引，可以获取到对应的类别名称。最后将标签信息和类别名称打印出来。

COding：

      # Example of a picture
      index = 25
      plt.imshow(train_set_x_orig[index])
      print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

![7](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/8a7fddbd-a178-4262-bdc9-ba12f5cfc00b)

深度学习中的许多软件bug来自于矩阵/向量维度不合适。如果你能够保持你的矩阵/向量维度的正确性，
你就能够在消除许多漏洞方面取得长足的进步。
### Exercise 1
Find the values for:
    - m_train (number of training examples)
    - m_test (number of test examples)
    - num_px (= height = width of a training image)

记住，train_set_x_origin是shape (m_train, num_px, num_px, 3)的numpy数组。
例如，您可以通过写入train_set_x_origin .shape[0]来访问m_train。

- m_train = train_set_x_orig.shape[0]：获取训练集的样本数量，即训练集的行数。
- m_test = test_set_x_orig.shape[0]：获取测试集的样本数量，即测试集的行数。
- num_px = train_set_x_orig.shape[1]：获取每个图像的高度/宽度，即训练集图像的列数（假设图像是正方形，所以宽度和高度是相等的）。

 Code

             #(≈ 3 lines of code)
             # m_train = 
             # m_test = 
             # num_px = 
             # YOUR CODE STARTS HERE
             m_train = train_set_x_orig.shape[0]
             m_test = test_set_x_orig.shape[0]
             num_px = train_set_x_orig.shape[1]

             # YOUR CODE ENDS HERE

             print ("Number of training examples: m_train = " + str(m_train))
             print ("Number of testing examples: m_test = " + str(m_test))
             print ("Height/Width of each image: num_px = " + str(num_px))
             print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
             print ("train_set_x shape: " + str(train_set_x_orig.shape))
             print ("train_set_y shape: " + str(train_set_y.shape))
             print ("test_set_x shape: " + str(test_set_x_orig.shape))
             print ("test_set_y shape: " + str(test_set_y.shape))

![8](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/edf556c3-286c-4728-b4f3-c153a23ad593)

**而现在，这些origin图都是3维数组的形式(num_px, num_px, 3) ，为了方便我们要reshape，test和train_dataset，变成（num_px  ∗  num_px  ∗  3, 1)..**
在此之后，我们的训练(和测试)数据集是一个numpy数组，其中每列表示一个扁平的图像。应该有m_train(分别为m_test)列.

重塑训练和测试数据集，使大小为(num_px, num_px, 3)的图像被平面化为形状为(num_px∗num_px∗3,1)的单个向量。
当你想要将形状为(A,b,c,d)的矩阵X平坦化为形状为(b∗c∗d, A)的矩阵X_flatten时，一个技巧是:

![9](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/8325807a-372d-4c7d-99c4-9174e66b4470)


在这段代码中，X是一个多维数组（或称为矩阵）。X.shape返回一个表示X的维度的元组，其中X.shape[0]表示X的第一个维度的大小，即行数。

X_flatten = X.reshape(X.shape[0], -1)将X重新形状为一个二维数组，其中X.shape[0]保持不变，而第二个维度的大小-1表示自动计算。
这意味着X_flatten将具有与X相同的行数，但是其列数将根据X的大小自动确定。

然后，.T操作符用于获取X_flatten的转置，即将行变为列，列变为行。最终，X_flatten被重新赋值为X的展平版本，并且已经转置了。
**Notice：注意在转至的时候，进行转至的数据是图像数据**

**Code**

            # Reshape the training and test examples
            #(≈ 2 lines of code)
            # train_set_x_flatten = ...
            # test_set_x_flatten = ...
            # YOUR CODE STARTS HERE
            train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
            test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

            # YOUR CODE ENDS HERE

            # Check that the first 10 pixels of the second image are in the correct place
            assert np.alltrue(train_set_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174, 213]), "Wrong solution. Use (X.shape[0], -1).T."
            assert np.alltrue(test_set_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145, 159]), "Wrong solution. Use (X.shape[0], -1).T."

            print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
            print ("train_set_y shape: " + str(train_set_y.shape))
            print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
            print ("test_set_y shape: " + str(test_set_y.shape))
            
![10](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/f951ef06-2e84-45fc-91d6-e49ca271e2d8)


**Expected Output**: 

<table style="width:35%">
  <tr>
    <td>train_set_x_flatten shape</td>
    <td> (12288, 209)</td> 
  </tr>
  <tr>
    <td>train_set_y shape</td>
    <td>(1, 209)</td> 
  </tr>
  <tr>
    <td>test_set_x_flatten shape</td>
    <td>(12288, 50)</td> 
  </tr>
  <tr>
    <td>test_set_y shape</td>
    <td>(1, 50)</td> 
  </tr>
</table>

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



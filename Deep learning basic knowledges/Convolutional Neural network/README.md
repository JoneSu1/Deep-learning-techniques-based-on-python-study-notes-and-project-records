# Week 1 Convolutional Neural network
## Computer vision(计算机视觉)

Computer vision is one of the areas that's been advancing rapidly thanks to Deep learning.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a88d3829-48c6-4585-96d5-b77d5f776a7a)

**In the Image Classification** We just need to recognize whether or not cats(0/1)

**In the Object detection** We need to identify the location of Objects in the image, 
and circle them in this image.

**Neural Style Transfer** We can have a neural network put them together to repaint the content image,
but in the style of the image on the right, and end up with the image at the bottom.

 如果数据量大的话，很难获得足够的数据以避免神经网络过拟合（overFitting）,同时对计算量和内存的需求是不可行的.

 而如何将大数据量的图片变得能够被训练，我们就可以使用到Convolutional Neural Network

 ## Edge Detection Example (边缘检测)

 The convolution operation is one of the fundamental building blokcs of a convolutional neural network.

 Using edge detection as the motivating example in this video.

 The early layers might detect the edge and then even later layers may detect the cause of complete
 Objects like people's faces in the case.

**而这个例子就是看如何detect edge**

![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/9174d947-6a8c-4c25-bb29-52d498f2d41b)


  1. 当拿到一张图片时候，第一件事就是detect the vertical edges(垂直边缘)，从图中看在vertical edges的过程中，人像被识别出来了.
  2. 然后第二步，就是进行 horizontal edges（横向边缘），这将检测到一些横向的背景.


**那么如何进行detect Vertical edge and Horizontal edges**

因为这是一个grey image所以只是一个6*6*1的matrix. 而不是 6*6*3，因为没有RGB channel.

in order to detect the vertical edges in this image, we can construct a 3*3 filter (过滤器).

然而我们需要做的就是通过3*3的filter对6*6的图像进行convolution operation. 在数学中"*"被认为是卷积计算符号，而在python中是元素乘法.

而在通过filter的处理之后，将得到一个4*4的 matrix.

计算过程是，将filter 复制到6*6的矩阵上面，然后进行element-wish operation.

就是将每个元素都相乘再相加，就可以得到4*4 矩阵的第一个element.

其中，filter中的数值是通过随机选值得到的。当然也有一些适用于特定情况的filter（如Sobel滤波器，Prewitt滤波器等）在进行detect edge的过程中是卓有成效的


**使用filter的过程**

![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/50a6a5e7-36cf-43a9-8043-db0d3c713832)


然后为了得到4*4中的下一个值，就需要将filter也同时向右边平移

 在numpy为workform的情况下，我们需要定义conv_forward的函数
 在TensorFlow为workform的情况下，我们可以使用既定函数：tf.nn.conv2d
 Keras: conv2D
 
 **下面的git图是在pooling过程中，filter怎么工作的**
![filter_Progress](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/e142fa6d-7e51-4b2e-bb48-947204e500d8)

**所以，为什么这是在进行detect Vertical edge**

这是进行解释的例子：

在左半边都是10的像素，右半边都是0的像素.
它的图就是左边更亮的像素强度，右边更暗的像素强度.
在这个图像的中间有一个明显的**分界**.

这个3*3的filter可视化之后就是一个由3色（白灰黑）组成的.更明亮的在左边，向右的暗.

然后再经过converlution operation之后得到的4*4 matrix，可视化之后是，
两边灰，中间白的。  这就和检测出来的vertiacl edge相对应，这个中间的高亮就是和6*6中的分界对应.
这表示刚好由一个强垂直edge再图像中间.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/31ff9c65-0a11-490d-ac06-3b4eed0a58ed)

## More edge detection

**Positive and negative edges**

还是和刚才的例子相似，但是第二个图改变了高像素的位置，这就导致4*4的哪个matrix中，出现了negative的像素值.

而再上面的图是，从暗到亮的边界，而下图就成了由亮到暗的了.

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c43e6fb4-420b-4932-b274-9941a6999828)

**以下是更多detect edge的例子**

**关于Horizontal edge**

不同的filter可以帮我们找到图中不同水平的edge.


在这个例子的右边，-30那里，取值的是很多0的地方，就是一个典型的负边界.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/215b1f65-9048-4d25-af00-5130756010a8)

**以下是介绍经典的filter数字设置的是，是根据vertical edge** 

**现在的介绍**

**如Sobel滤波器，Prewitt滤波器等**

**如sobel filter**： 优点在于给中间行赋予了更大的权重。 就可能让它更加的稳定.

**Scharr filter**

**如果想要detect 一些复杂的edge of image**：
并不需要计算机视觉的研究人员挑选出这9给矩阵元素.
可以将这9个数给定义成parameter W，并通过backward propagation来得到他们的数值，从而得到一个优秀的filter.

**如果需要也可以尝试调节element到达进行45°，70°，73°无论什么角度的学习**

![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d68c4542-97f9-4699-b9c9-8f0c584d510b)

## Padding（填充）

构建CNNs 网络的基本操作就是是Padding.

**关于Padding过程**

如果我由n*n的input matrix， f*f的filter matrix，最后的输出，我们会得到的是（n-f +1 ）* (n-f+1) 的matrix.

**进行padding的缺陷（downside）**

1. 如果每一次我使用一个卷积操作，图像就会缩小（shrinking output）（可能做不了几次图片就会变得非常小）
2. 图片中的 pixel(像素) of edge和corner只会在输出中被使用一次，然而位于中间的pixel会在多个filter上重叠.相对来说pixel of edge and cornor被使用次数会少很多（所以会损失很多边界信息）.
   
So to sovle both of these problems,both the shrinking(收缩) output.
我们可以在使用coverlution operation之前要执行**Padding**.

 我们可以pad image到那个input matrix里面.
 如下图，我们可以用一个额外的边缘（border）填充图片（一个pixel大小为1的额外边缘），来让

做完padding之后，本来的input matrix是6*6现在就变成了8*8

所以这个output再进行了padding之后就不再是4*4了，而是8-3+1，8-3+1=6*6的matrix了.

 **当我用0来padding的时候**

如果 p 是填充的数量，意思就是填充的圈数.
p =  padding = 1 ,此时就变成了（n + 2p - f+1）,(n + 2p - f + 1)
                             6 + 2 -3+1, 6 = 6*6的matri

![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/b88b1059-1a16-4f81-a1c7-3ca6a8487755)

**那到底需要padding多少呢**

**一般就两种卷积方式：Valid 卷积和 Same卷积**

**Valid（有效）卷积**：意思是没有padding，就是n*n 卷积 f*f 得到n - f + 1, n-f+1

**Same convolution** Pad so that output size is the same as the input size.
就是在padding之后，input和output的matrix的大小是一样的， n + 2p - f+1 = n.
则p = (f-1)/2 ,所以当filter是奇数的时候，就可以用这个公式使得输入和输出相同size的matrix.

**而在computer vision中基本上filter都是用奇数（odd）而不是偶数（even number）**

如果filter is even number, you may need some asymmetric（不对称的） padding(不对称填充)
只有f是odd时候才能有same convolution.

而且当是odd filter时候，会在中间有一个特殊的pixel点，这在vision learning中是好的.

## Strided Convolutions（滤波器）

**Strided convolutions**是一种在卷积神经网络（Convolutional Neural Networks，CNNs）中常用的技术。在卷积操作中，"stride"是指滤波器（或称为卷积核）在输入数据上移动的步长。例如，stride为1意味着滤波器每次移动一个像素，stride为2意味着滤波器每次移动两个像素，以此类推。

Strided convolutions常常被用作下采样（downsampling）的手段，以减少网络中的参数数量和计算量，同时也能增加模型的感受野（receptive field）。这在某些任务中是有用的，例如在图像分类中，我们可能更关心图像的全局信息，而不是局部细节。

使用strided convolutions可以减小输出的空间维度（例如，宽度和高度）。例如，如果你在7x7的input matrix上使用3x3的filter和stride为2的卷积，那么输出将是3x3的。这是因为滤波器每次移动两个像素，所以它覆盖的输入区域更少。 
我将stride seting成 2意思就是，每次convolution operation 都要hop over 2 steps.
就是从第一个pixel到下一个重合的pixel之间由两个单位的差距，不管是横向还是纵向.

n*n 卷积 f*f 那么output ，end up ： (n + 2p -f)/S  + 1 * (n + 2p -f)/S  + 1就是output的size.

padding p   stride：  S，  S=2 ， （7 + 0 - 3）/2 +1=3，所以size of output是 3*3

![10](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/ccd276ac-d706-454e-9572-0997a6156a8c)


**特殊情况这个分数不是整数的时候**

Which is one of this fraction(分数) is not an integer(整数).

In this case, we are going round this down. (round down 向下取整，4舍5入)

[Z] = floor(Z) 意思就是对Z进行round-down
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/4eb8d88a-fc00-4243-9d4e-0b6fda0a3984)

**Technical note on cross-correlation vs. convolution**

一般都会跳过（如果有必要可以单独来看这个技术部分）
对filter进行操作.
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/acc6530e-e8fd-4b2e-83da-eb4b405829d4)

flip it on the horizontal as well as the vertical axis.
基于横轴和纵轴进行翻转.

然后得到mirroring it both on（两个的镜像）

所以它的结论就是：A于B卷积后再和C卷积 = B和C卷积之后再和A卷积. 这对于某些信号处理有用（对深度学习用处不大）

## Convolutions Over Volume(卷积在体积上)

### Convolutions on RGB images
就是不止对grey image处理了，而是Three-dimension的处理，也就是RGB images（red，green，blue的3通道图像）

进行Convolution operation 的过程还是相似的，只是input变成了n*n*3并且filter 也变成了f*f*3

**图中的，6*6*3**：第一个6是代表hight，第二个6代表width宽，3代表channel number.
input的channel和filter的channel数量必须相同.

 进行Convolution operation 的过程和1 dimension时候是相似的，但是是一整个的square放了上去.

这个3*3*3filter是一个3的立方体（cube）.

 为了得到下一个数字，必须将这个cube移动一个单位去计算27次相乘，再相加.

 ![filter_Progress](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/e1e6667a-280a-474e-a8f9-fad342455399)

 
如果我们只想对red channel进行vertical edge的convolution operation就可以将filter的第二和第三层都设成0.
如果不在意vertical edge是属于哪一个channel可以都使用相同的filter层.
通过设置parameters of filter将会能得到达成不同效果detected edge.

**In by computer vision**: 通常当你输入一个certain（固定）height和certain width，以及certain number of channels的input，我们会设置一个固定channel和不同宽高的filter，我们可以由一个只负责red或者green或者blue通道的filter。

![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/f5aeea3d-2cf4-43ad-bed5-531ff73d024a)

**The one last important concept**

**如果我们不止想进行vertical edge detection 还想同时进行horizontal edge detection或者其他角度的detection**

**如何同时应用多个filter**

下图中黄色为一个vertical filter，橘色为第二个Horizontal filter.

在使用这两个不同的filter之后得到了两个4*4 的matrix，我们可以吧第一个filter的结果放在第一层，把第二个的filter的结果放在第二层，使得我们的output变成了4*4*2的output.

**Summarize：** n * n*nc  卷积 f*f*nc 得到一个 n-f+1,n-f+1, (number of filters).

卷积是很强大的，这个filter的设置让我们可以同时的处理vertical，horizontal以及更多的不同特征.

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/aae347f7-4eb7-4c75-a05e-13a3c78b5016)

# One layer of a convolutional network

例子：对卷积完之后的结果加上一个常数集b并且用一个非线性的函数进行处理（比如ReLU）.
然后将得到的几个filter的结果整合到一起.

然后将这个处理Convolution network的步骤和之前neural network的步骤对应起来。

其实filter的处理是通过相乘在相加得到结果，是一个线性方程计算的结果，而在构建卷积神经网络（Convolutional Neural Networks，CNNs）时，非线性函数（通常被称为激活函数）的使用是至关重要的。这是因为它们赋予了网络捕捉复杂和非线性问题的能力。如果没有非线性函数，无论网络有多深，它都只能表示线性函数，这大大限制了网络的表达能力。

所以这个a[0](input)到a[1]的步骤：
1. 线性计算得到filter output的结果
2.  然后添加bias再通过非线性函数RelU进行操作得到a[1]

**Number of parameters in one layer**

If you have 10 filters that are 3*3*3 in one layer of a neural network,
how many parameters does that layer have? 

其中一个filter的3*3*3 + bias = 28个，而有10个filters则是28*10=280个parameters.

这种复杂的数学组成使得再convolution neural network中不太容易overfitting.
这意味着我们训练出的filter模型，可以应用到非常大的图像（特征detection）中.

![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/3aca8e44-458e-4246-bf8d-c5b916368f6f)

**Summary**
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/fba3b6f7-691a-4a6e-b509-db838863a53f)


## A simple convolution network example

Example ConvNet:

一组图片判断是不是猫，input是39*39*3 那么就意味着

        H[0] = 39
        W[0] = 39
        n[c] = 3
        f[1] = 3
        s[1] = 1
        p[1] = 0
        10个filter
        
所以output层的是(n+2p-f)/s +1 得知output是37*37*10（有10个filter）

       nH[1] = nW[1] = 37
       nc[1] = 10

而下一层是:  

       f[2] = 5
       s[2] = 2
       p[2] = 0
       20个filter
       
所以output层是（n+2p-f)/s +1 = 17, 所以是 17*17*20

       nH[2] = 17
       nW[2] = 17
       nc[2] = 20
       
再加上一个卷积层：

       f[3] = 5
       s[3] = 2
       40个filter
所以output layer是（n+2p-f）/s +1 = 7, 所以是7*7*40

所以我们的3次卷积将39*39*7的图像給计算出成了7*7*40=1960的特征.

然后我们将这1960个单元进行reshape的转换，变成one dimension array（vector）.
（all 1960 numbers, and unrolling(展开)them into a very long vector.）
然后导入logical的binary classification函数Sigmoid或者是多分类的Softmax，得到判断结果.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/355c2542-1df4-43c4-9293-c40067be15fd)

**Types of layer in a convolutional network**

- Convolution(CONV)
- Pooling(POOL)
- Fully connected (FC)

 ## Pooling layer

 ConvNets often also use pooling layers to reduce the size of the representation,
 to speed the computation, as well as make some of the features that detect a bit more
 robust(强壮的).

 在卷积神经网络中，池化层通常在卷积层之后使用，主要有以下几个原因：

1. 降维：池化层减少了输入数据的空间尺寸，从而减少了网络中的参数和计算量，提高了计算效率，同时也有助于防止过拟合。

2. 平移不变性：池化层可以提供一种形式的平移不变性，即输入数据稍微移动，大部分池化输出的值不会改变。这对于图像识别等任务非常有用。

3. 特征提取：池化层有助于提取主要的局部空间特征。

4. 模型正则化：通过减少参数数量，池化层也有助于防止模型过拟合。

而再现有的pooling 的方法中，多用的是：**Max pooling** 和 **average pooling**.
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/6fac6eb3-3f28-425c-adf6-81acdbfb4ba8)

**Max pooling**

**在实践中发现这个方法效果更好一些**
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/b970bcd0-12ec-4ff2-adb2-ba1ebb3436b8)

我们将经过Convolution operation处理的output数据分成4个颜色的区，然后取每个颜色中最大的数字.

其实就是设置了一个filter，f = 2, s= 2,但取的是最大值.
如果把4*4看作是feature的集合的话，就是再神经网络某个层中的激活状态中的一个大的数字，意味着它或许检测到了一个特定的特征，显然它在左上角那里有一个大的数字9，这个特征也许他不是cat eye detect，但是在右上的区域没有这个特征。 

所以Max Pooling做的是，检测到所有地方的特征. 如果在滤波器中任何地方检测到了这些特征，就保留最大值.

**特点**
- 它有一套Hyperparameters： f =2, s=2
- 但是在实际上没有任何需要梯度相加算法学习的东西（这就表示，只要确定了f和s，怎么迭代都不影响它）
- 因为Max pooling实际上也是用的filter做的，所以同样可以用n = （n+2p-f）/s +1来算

![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/5f8bf68c-7db1-4de5-b7d3-fd3cb48bdfce)

**关于Average Pooling**
**一般是选择用Max pooling**

和之前的Max pooling差不多，就是在设置filter的时候，让它不是取最大值而是取平均值.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/206f1412-fe9e-40a9-b237-d1a67fadf1b5)

## CNN example

在进行Convolutional Neural network 构建中，将CNN1和PooL1看成是一层. 

进行了3次convolution operation 得到400*1的verctor， 就是被unrolling的neural unite.
然后用我们400*1的做为input作为输入，在下一层layer中构建给120 unite的全连接层（FC3）和普通的单层神经网络层相同所以这层是（120，400），然后再构建一个FC4(84,120)。然后在下一层中使用Softmax作为activation function就可以得到进行识别了的数据。

**关于Hyperparameters的设定，尽量取找别人在文献中使用的，别自己造**
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/cc7359f6-322d-4b79-82bb-161479805c3a)

**一些总结**
随着网络的构建，Activation size是在逐渐减小的.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/413e8529-2833-4d38-8def-549cc4c9dd17)

## Why Convolutions?

**Advantages**
1. parameter sharing（一个filter在经过1层所有数据时候都是相同的）
2. sparsity of connections(连接稀疏)，输出的结果和周边的结果无关.
 
 参数共享有以下几个主要优点：

减少参数数量：通过在整个输入图像上共享同一组权重，我们大大减少了模型的参数数量。这使得模型更加高效，同时也减少了过拟合的风险。

平移不变性：由于同一组权重在整个图像上都被应用，所以无论特征在图像中的何处，CNNs都能够识别出它。这就是所谓的平移不变性，它使得CNNs非常适合处理图像数据。

学习空间层次结构：通过在不同的层级上应用参数共享，CNNs能够学习到图像的空间层次结构。在较低层级，网络可能会学习到边缘或颜色斑块等简单特征；在较高层级，网络可能会学习到更复杂的特征，如物体部分或整个物体。

**Advantages**

**参数共享**：在CNNs中，同一组权重（即卷积核）在整个输入图像上都被应用。这种参数共享策略大大减少了模型的参数数量，使得模型更加高效，同时也减少了过拟合的风险。

**平移不变性（Translation Invariance）**：由于同一组权重在整个图像上都被应用，所以无论特征在图像中的何处，CNNs都能够识别出它。这就是所谓的平移不变性，它使得CNNs非常适合处理图像数据。

**局部感知（Local Perception）**：CNNs的每个神经元只与输入数据的一个小区域（即其感受野）相连，这使得CNNs能够捕捉到图像的局部特征。

**层次结构（Hierarchical Structure:）**：CNNs通常由多个卷积层和池化层堆叠而成，这使得网络能够学习到数据的层次结构。在较低层级，网络可能会学习到边缘或颜色斑块等简单特征；在较高层级，网络可能会学习到更复杂的特征，如物体部分或整个物体。

**在图像和视频处理任务上的优秀表现（Excellent Performance on Image and Video Processing Tasks:）**：CNNs在许多图像和视频处理任务上都表现出色，包括图像分类、物体检测、语义分割、人脸识别、行为识别等。

**如何训练一个好的模型**
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/3e4684da-09a7-4e46-822f-69d398b5922e)

# About the case studies

借鉴别人的模型和代码，但是我们有不同的计算版本任务，我们需要把别的的arcticture apply that to my problem.

## Outline
这些是pretty effective network
### Classic networks:
1. LeNet-5
2. AlexNet
3. VGG

#### ResNet（152 layers）
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/3debf249-df4f-4ea1-ae80-4b6bfddb4d3b)

这个LeNet - 5是识别文字图像的经典CNN模型. 现在已经不用了.

#### AlexNet

这个也是用足够多的filter把image 给分隔成足够多的dimensions.

这个模型和LeNet-5很像，但是用的是Max pooling，和ReLU函数，并且这个模型更大.

这个的文章的核心是在GPU算力低的时候。所以他们使用了两个GPU，将网络中很多层被分割到两块不同的GPU上.
但现在不是很重要了.

 
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/70bceae3-222c-426d-ab6c-3881baa1c829)

#### VGG - 16

VGG-16 的结构更简单，更关注卷积层

是一个简洁且好用的模型，是随着length and width降低而深度加深的一个模型.

![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/f07addd5-be85-4e21-829f-dbd786a9643a)

### 更先进和有力的神经网络结构

#### 残差网络（Residual Network）

跳跃连接（skip connection） 能让我从一层中得到激活.

并且他解决了Vanishing and exploding gradient 出现的问题.

而利用skip connection能让我懵训练网络层次很深的残差网络.

在这个网络中，我们需要设置Residual block

**Residual block**


残差块 (Residual Block)

残差块是深度学习中 ResNet（残差网络）的核心组件。ResNet 的主要创新是引入了所谓的“残差连接”或“跳过连接”，这些连接可以直接将输入传递到后面的层，从而避免了深层网络中的梯度消失问题。

残差块的基本思想是：对于一个深层的神经网络，如果我们假设某些层应该近似于恒等函数，那么让这些层学习一个残差映射（即与恒等函数的差值）可能会更简单。

具体来说，考虑一个输入 x 和一个深层网络的输出 H(x)。在 ResNet 中，我们不直接学习这个输出 H(x)，而是学习残差 F(x) = H(x) - x。之后，我们可以简单地通过 F(x) + x 来获得 H(x)。

这种结构在实践中非常有效，因为“跳过连接”提供了一个直接的途径，使梯度在反向传播时可以绕过一些层，从而避免了梯度消失问题，使得网络可以安全地增加深度。

在实际的残差块设计中，这个块可能包含几个卷积层、激活函数和批量归一化层。残差连接确保了块的输入可以直接添加到块的输出。

总的来说，残差块通过引入跳过连接，使得深度神经网络的训练变得更稳定和高效，从而允许我们构建非常深的网络结构。



**从a[l]到a[l+2]的这个信息，需要经过下列的过程，这被叫做这层组成的主路径**

而shotcut 是指再a[l] 到a[l+2]的过程中，在进入线性函数之之后，在进入ReLu之前，跳过部分主路径，直接得到a[l+2].

这个shot cut最大的差别在：

1. 在经过主路径时候的公式是： a[l+2] = g(Z[l+2])
2. 在shot cut路径中的公式是：a[l+2] = g(Z[l+2] + a[l])

这就使得这成为了Residual block

通过大量堆叠Residual block 形成一个深层网络。


**在plain network的训练中，不管是用gradient descent还是别的来进行，都会在layers大到一定数量就发现training error会反弹**

而这限制我们构建更深的网络，因为网络约深可能误差约大.

**而如果我们使用了，Residual network，通过那些skip layer，以及shot cut能让我们的trining error不发生反弹**

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/256d30db-ccd3-4294-884e-266e33ee07d8)


### Why do ResNets work so well?

Why do residual networks work?

由于我们加上了一个shot cut的block

所以我们a[l] 输入到Residual block之后得到的是a[l+2]

这意味着Residual block更容易学习恒等函数





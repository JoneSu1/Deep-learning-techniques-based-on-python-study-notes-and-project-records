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





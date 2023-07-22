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


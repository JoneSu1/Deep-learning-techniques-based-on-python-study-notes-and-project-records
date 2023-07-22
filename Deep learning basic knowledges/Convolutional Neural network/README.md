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
  3. 


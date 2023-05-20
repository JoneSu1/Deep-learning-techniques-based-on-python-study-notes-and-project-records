总览

**Deep learning models for Genomics**

    Supervised learning 有监督学习
      Sigle-layer, Multi-layer NN
      Training neural networks for supervised learning
      DNN, CNN, RNN, GCN appying samples
      Single-task, multi-task, Multimodal, Transfer Learning
      
------------------------------------------------------------------
    Unsupervised learning 无监督学习
      自动编码器AutoEncoder应用举例
      生成对抗网络GAN的应用举例
      
 ![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d3f46816-6ace-4e8b-8596-9fb9929112ec)
   
    **深度学习模型的分类**
    •
    根据模型构建的方法，按照研
    究的问题可以分为：有监督和
    无监督的。
    •
    有监督学习是指一类针对有标
    签的数据来预测无标签数据的
    标签的算法。如果我们把连续
    数值变量也视为标签的话，那
    么回归也是有监督学习。例：
    分类问题。
    •
    无监督学习是指一类针对无标
    签的数据进行规律发现的算法。
    例：聚类问题。
--------------------------------------------------------------------------

    监督学习
    •
    监督学习问题的一个例子是：给定
    RNA 上的特征，例如规范剪接位点序列的存在与否、
    剪接分支点的位置或内含子长度，预测内含
    子是否被剪接掉。训练机器学习模型指的是
    学习其参数，这通常涉及最小化训练数据的
    损失函数，从而对看不见的数 据做出准确的
    预测。
-------------------------------------------------------------------------
**如何用单层神经网络（single layer）解决上述问题**
    ![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/4e0862d5-42bc-47b1-837f-760c449f3ee7)

    我们在分类特征确定时，将有碱基的定为数字1，没有碱基的定为数字0. 以1，0为特征进行模型训练。

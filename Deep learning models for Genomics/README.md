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

    我们在分类特征确定时，将有剪切点的定为数字1，没有剪切点的定为数字0. 以1，0为特征进行模型训练。 在二维空间中分类的分布表现在Branchpoiont. 使用单层网络时候，每一个input都和output直接相关.
    （可以根据定义，来研究RNA上的特征）就可以构建出一个能进行筛选的modle.
    
**如何使用多层神经网络（Multilayer neural network）（Fully connection network）解决上述问题**
    ![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/3047e686-a164-44d8-bdc2-d8b9cdb23fc8)

     使用多层神经网络时，每个input会和共同影响Hidden layer中的每一个因素， 而Hidden layer中的因素和output直接相关. 最后的分类结果是成线性的，并区分开了.
     
 **监督学习如何训练神经网络？**
 
    •
    数据集划分和确定预测目标。
    •
    数据集分为训练集：用于优化模型参数；验证集：用于评估模型性能；
    测试集：用于对最佳开发模型的最终评估。预测的准确性通过不同的评
    估指标来衡量。
    •
    使用训练集拟合参数。（是用来调整模型的）
    •
    首先对神经网络的参数进行随机初始化，然后使用随机梯度下降或其变
    化的方法进行迭代优化。（拉格朗日等算法，用来寻找最大或最小值的）
    •
    使用验证集选择超参数（得到的这些参数集合是否适合于筛选结果）
    •
    通过评估损失或验证数据集上的评估指标来训练过程。
    
---------------------------------------------------------
**训练的过程**
![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/3173af49-ec0b-49e8-ae32-61bb020dc9a9)

     
    通过把训练集中的一部分数据（Batch）投入神经网络进行训练，然后得到一组参数. 我们根据过程中的loss函数值来判断参数的价值. 
    然后根据loss值和超参数来确定最佳参数组，其实就是取在训练集和验证集中loss值都最低的那一组参数. 具体看图C. Validation（有效组，验证集）
    验证集主要是为了防止过度拟合，导致偏离目的.
    
 ---------------------------------------------------------
 **主要的神经网络介绍**
 
    一. 全连接神经网络（DNN）（主要用于确定突变位点的调控功能，从是否剪切，是否致病等特征进行）
    全连接神经网络已被用于许多基因组学应
    用，其中包括根据序列特征（如剪接因子
    结合基序的存在或序列保守性）预测特定
    序列中剪接的外显子百分比；优先考虑潜
    在的致病基因变体；以及利用诸如染色质
    标记、基因表达和进化保守性等特征来预
    测给定基因组区域中的顺式调控元件。全
    连接层构成了深度学习中必不可少的构建
    块，可以与其它神经网络层 如卷积层 有
    效结合。
    
 --------------------------------------------------------------
  
    二. 卷积神经网络（CNNS）(最初是用于图像识别)（在基因组研究中，主要是用于确定局部特征的，比如TF结合位点）
      自卷积神经网络最初应用以来，仅以
    DNA 序列为基础就被应用于各种分子表型预
    测，成为最新的前沿模型。应用包括转录因
    子结合位点分类和预测分子表型，如染色质
    特征、 DNA 接触图、 DNA 甲基化、基因表达、
    翻译效率、 RBP 结合和 microRNA miRNA
    靶点。除了从序列中预测分子表型外，
    CNNs 已经成功地应用于传统上由手工制作
    生物信息学流程处理的更多技术任务。例如，
    它们已被用于预测导向 RNA 特异性、去噪
    ChIP seq 、提高 Hi C 数据分辨率、根据 DNA
    序列预测实验室起源和调用遗传变异。

**例子：如何用局部特征解决问题.（转录因子结合位点预测）**

![6](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/390ea745-9cf5-48c0-82ae-4ce7cca7c7dd)

     由于基因片段的特殊性，碱基只有4种出现的可能，A（腺嘌呤）,C（胞嘧啶）,T（胸腺嘧啶）,G（鸟嘌呤）.
     所以在input的阶段，将4种碱基出现的情况用热图的形式表现出来。 这让我们能根据像素的改变来确认碱基的排列. 
     因为CNN网络并不是识别所有像素点，而是区域性识别。 所以将包含TF位点信息的碱基们作为一个模块（6*4），而由于我们的目标是两个，
     所以将会有两个通道进行卷积.
     ![12](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/49ee32ef-e0c5-4cad-8a2c-e10c70f4fe36)

 ------------------------------------------------------
     而在Hidder layer层中包含了两个通道CATA1和TAL1（两个通道的算法有差异），又将之前包含TF位点信息的整个模块（6*4）
     看作是单独的一个像素点分别在两个通道中表示出来，然后激活其他模块的基因序列，最后进行池化.
     

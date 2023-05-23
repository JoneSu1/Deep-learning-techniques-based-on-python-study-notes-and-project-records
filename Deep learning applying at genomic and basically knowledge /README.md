**深度学习在下一代测序中的应用**
![25](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/cb24097c-963e-4c7a-a3d6-df161da49d5b)

      深度学习能解决以下三大类的问题：
      1. Regulation of gene expression
      2.Genome analysis and SNP research(突变位点确定)
      3.Early Detection of cancer
      
      
**为了解决以下三类问题，每一个问题都被进行了拆分**
**Regulation of gene expression**
     
        在基因表达调控问题探索中，我们会从以下角度进行.
        1.Epiggenetic modification(寻找富集的点，寻找进行修饰的点)
        用到的深度学习模型：Deepmethy1 和 DeepChrome.
        2.Proteins and regulatory sequence(预测相互作用和调控,基因是否binding到某一段核酸上等)
        用到的深度学习模型：Basset, Deep motify, DanQ, PEDLA, DECRES, DEEP, DeepBind
        3.Prediction of splice variants mRNA
         用到的深度学习模型：DeepSlice
         
         
  **Genome analysis and SNP research**
  
  
         在碱基突变研究中，以下角度进行：
         1. SNPs in in-coding and non-coding regions of genome
         模型：DeepSea, Diet Networks, DANN
         2. Protein structure prediction
         模型：PconsC2
         
  **Early Detection of cancer**
        
        
          主要用于区分肿瘤分型的.
          DeepType


**基因组常用深度学习框架**

            •1.安装并介绍深度学习工具包 tensorflow,keras pytorch
            •2.在工具包中识别深度学习模型要素
            •2.1.数据表示
            •2.2.张量运算
            •2.3.神经网络中的“层” （每一次层是怎么定义的，每一个层有多少个节点）
            •2.4.由层构成的模型（这些层是要构成什么样的个模型）
            •2.5.损失函数与优化器（如何利用loss函数达到最小，并找到最优模型参数，可能会用梯度下降等作为优化器）
            •2.6.数据集分割（训练集合，测试集，验证集怎么分割，才能保证最优）
            •2.7.过拟合与欠拟合
            
**深度学习在生物信息中的基本流程**
![26](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/7e7a8d32-0083-4a89-89ed-feb6db5a3c8f)

            •设计实验，收集数据
            •数据清洗
           （在拿到原始数据后，进行预处理，例如质控）将得到clean data（input数据）
            •特征选择
            （可以用统计学方法找，也可以用深度学习方法找）最后得到和feature有关的向量数据
            •模型构建
             （把上一步得到的向量数据导入模型中，去训练模型并得到最佳参数模型）
            •模型评估
            （进行模型效果测试）
            
**基本数据**

            •序列数据（最常用，可以导入到list格式中）核酸序列数据，蛋白的氨基酸数据.
            •矩阵或者张量数据（多个基因表达的数据（矩阵），具有时间差异和印象的表达数据（张量））
            •成像数据（常见）
            
 **如何处理序列数据**
            
 ![27](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/8bd557ec-4966-43eb-bc25-351a8e8f49c9)
           
             如何解读DNA序列呢？
             1. 我们可以对DNA序列进行注释，将这个数据和临床等联系起来（主要就是把数据中普适性的特征给标记出来）可以确定那一个片段是
             内含子还是外显子，是否是coding的片段。
             2. 可以看到在序列的起始端和结束端直接，包含着内含子（intron）和外显子（exon）并且他们会相互作用，他们越多，序列越长.
             
             关于蛋白数据：
             1. 得到的是20种氨基酸的组成的，并且需要考虑蛋白的3D结构。 在进行注释的时候难度大.
            •最基本的生物数据之一，通常为 DNA 序列， RNA 序列，蛋白质序列。 在人类基因组计划早期的问题是，如何
            快速进行基因组注释，该问题可以表示如下：
            •基因组注释是一个有监督或者半监督的问题，因为一段序列是不是基因可以通过EST(表达序列标签)来判定，其他特征可以通过一些生化
            或者分子实验来标定，所以我们可以得到数据标签。     


**矩阵数据**
       
![28](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/0bf9e4fe-9e23-4461-b0c8-d11b9aa50f21)
 
            •芯片技术和后续的高通量测序技术带来了很多种矩阵数据 这类矩阵通常是对某类型生物特征
            （基因，蛋白，表观修饰，染色质互作）的丰度汇总而成的。最典型矩阵数据是基因表达谱，
            基因表达谱矩阵可以通过 RNAseq 数据进行比对后的转录本定量产生。
            
**成像数据**

![29](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/9ce6e177-15f2-47b8-a14e-22d526f318a7)

            •成像数据表达更多的是生物体内部空间位置（还有形状或者结构）的信息。例如，
            一张蛋白亚细胞定位的图像，可以反映某标记的感兴趣的蛋白质位于细胞中的什么位
            置，如果我们有很多这样的图片，常用的方法是先标记一部分数据，训练一个卷积神经网络，然后再对剩下的图片进行预测。

**数据集和划分**
            
            •在机器学习算法中，我们通常将原始数据集划分为三个部分（划分要尽可能保持数据分布的一致性）：
            （1 Training set （训练集 ）: 训练模型
            （2 Validation set （验证集） 选择模型
            （3 Testing set （测试集） 评估模型

--------------------------------------------------------------------
            
            •数据集较小
            •如果数据集较小时，一般采用简单交叉验证的方法，即不设置验证集，而只设置训练集和测试集， （
            就是把全部数据都拿来做训练，把全部数据分成不同比例，进行多次训练，然后取多次训练的平均结果作为结果）
            例如第一次取7成做训练，3成测试，第二次就9成训练，1成测试，第三次就8成训练，2成测试。
            训练集和测试集的比例设置一般为 2:1 ~ 4:1 。根据目前我所看到的方法，大多数人将比例设置为7:3 。
            •数据集较大
            •如果数据量较大时（数据集以万为单位），一般训练集、验证集、测试集的分配比例为 6:2:2 。
            •数据集非常大
            •如果数据量更大时，例如百万级的数据集，一般划分比例在 98:1:1 以上（即根据情况再提高训练集的占比）。
            
            
**什么是神经网络中的层**
**1.激活层**

![30](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/6c085e11-2644-44b2-a3b7-96227e223fd4)

              在神经网络中加入激活函数存在的意义就是为网络增加非线性因素，比如全卷积神经网络，不加入激活函数，就是一堆线性操作矩阵运算，
            对于复杂任务仅用线性操作而不用复杂的函数表示肯定效果不会好。
            卷积层、池化层和全连接层都是线性的，所以，我们要在网络中加入非线性的激活函数层。一般一个网络中只设置一个激活层。
            •所谓激活，实际上是对卷积层的输出结果做一次非线性映射。激活函数可以引入非线性因素，解
            决线性模型所不能解决的问题。从上图中可以看到，输入信息 x在神经元内首先经过加权求和，
            然后通过激活函数的非线性转换，将数据控制在一定范围区间内。转换的结果作为下一层神经元的输入，或作为结果进行输出。
            •常见的激励函数：sigmoid 函数、tanh 函数、 ReLu 函数、 SoftMax函数、 dropout 函数等。
                  如图所示，进入到激活函数之前，是使用了加权算法的全连接神经网络，output结果进入到激活层之后，就会
                  根据所选的激活函数例如Relu，对输出的结果进行限制，只有正数据（保留对feature有意义的，筛除没有意义的）
                  
 **2.池化层**. 
    
 ![12](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/23fb549b-e9ad-4ef2-a2d2-e5eec3db34e1)
       
   ![31](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/0aae5167-52fc-4839-ba16-26811c7f5ac4)

               1.池化是为了把数据降维（压缩），改善结果. 我们需要找出一个普适性的规律，不能让我们的模型过于适应这个数据.
            过于适应会发送过拟合。 而我们通过设置池化，通过选取一个小区域的最大值或者平均值的方法，重新构建数据来达到目的.
               它在卷积后再使用可以到达降采样的效果
            
            2.池化层也称为抽样层他的扫描方式和卷积相似，计算的方式不再是扫描框与被扫描的区域点乘，池化运算方法有以下 3 种：
                  均值池化：提取被扫描区域中有所值的均值；
                  随机池化：提取被扫描区域中的任意一个值；
                  最大池化：提取被扫描区域中值最大的一个。
                  优点：
                  可以压缩数据，改善结果，是神经网络不容易发生过拟合 。和卷积层一起使用，有效的减少
            数据量，起到了降采样的作用，相当于对卷积后进一步的特征提取与压缩。
            
            
  **卷积层**
  ![32](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/776fb442-6114-4250-966f-872898c2552b)
![33](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/f55d9139-1886-4a5b-8480-cfaf16ed3afe)


                 卷积是为了得到特征map。池化是为了将特征控制在一定范围.
                 而卷积过程是使用卷积算法，对原始图片或者数据进行扫描。然后计算并，把特征数据筛选了出来构成了新的feature map.
                 而在卷积过程中进行计算的是，卷积核（convelutional kernels）
               卷积神经网络中每层卷积层（Convolutional layer ）由若干卷积单元组成每个卷积单元的参数都是通过反向传播算法
            最佳化得到的。卷积运算的目的是提取输入的**不同特征(局部特征)**，第一层卷积层可能只能提取一些低级的特征如边缘、
            线条和角等层级，更多层的网路能从低级特征中迭代提取更复杂的特征。
              在卷积神经网络中，卷积运算是对两个矩阵进行的。如下图，左侧为输入矩阵 M中间为过滤器 F （也叫卷积核 F 以一定步
            长在 M 上进行移动，进行点积运算，得到右侧的输出矩阵 O 。这个就是卷积神经网络中卷积层最基础的运算。在实际的操作中，还
            存在一些额外的操作。
            
**全连接层**

![34](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/866decf1-328f-4ede-83fd-66ccddddfa80)


                 **在多次使用卷积和池化后，我们留下的特征数据会越来越少，到了最后，我们需要把这些特征数据和结果联系起来，
               于是就有了全连接网络，这些每一个筛出来的特征都和结果的一个结论有连接**
          四、全连接层与输出层
              全连接层（fully connected layers FC ）在整个卷积神经网络中起到“分类器”的
            作用。如果说卷积层、池化层和激活函数层等操作是将原始数据映射到隐层特征空
            间的话，全连接层则起到将学到的“分布式特征表示”映射到样本标记空间的作用。
            全连接层之前的作用是提取特征，全连接层的作用是分类。
             输出层：就是完成了对之前特征的分类，符合这些特征的输出的话，有多少概率是符合我们的筛选目标的.
             
 **在训练出模型之后，我们要根据训练的结果来确定是否是最佳的模型**
 **损失函数可以帮我们确定模型最佳参数**
 
 

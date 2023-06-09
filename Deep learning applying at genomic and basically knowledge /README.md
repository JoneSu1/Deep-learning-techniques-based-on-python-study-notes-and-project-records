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
 
 ![35](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/56c0a521-736f-425d-81a0-c9f080125367)

            关于损失函数的选择，我们需要根据真实值（y）和预测值（y）的数据分布来确定。 是离散型？线性性？
           **损失函数概念**
              损失函数用来评价模型的预测值和真实值不一样的程度，损失函数越小，通常模型的性能越好。不同的模型用的损失函数一般也不一样。

 **损失函数的作用**
 
                  **前向是做预测，得出预测值。 后向是参数调整**
              •损失函数的使用主要是在模型的训练阶段，每个批次的训练数据送入模型后，通过前向
            传播输出预测值，然后损失函数会计算出预测值和真实值之间的差异值，也就是损失值。
            得到损失值之后，模型通过反向传播去更新各个参数，来降低真实值与预测值之间的损
             失，使得模型生成的预测值往真实值方向靠拢，从而达到学习的目的。
             
**损失函数的分类**

            •损失函数一般分为4 种：
            1.HingeLoss 0-1 损失函数：感知机（SVM）就是用的这种损失函数；
            2.对数损失函数：逻辑回归的损失函数就是log 对数损失函数；
            3.MSE平方损失函数：线性回归的损失函数就是 MSE
            4.Hinge损失函数： SVM 就是使用这个损失函数；
            5.交叉熵损失函数：逻辑回归的损失函数用 sigmoid 作为激活函数，常用于二分类和多分类问题中。（用的多，离散型都用这个）
            
**如何选择损失函数**
            
              •通常情况下，损失函数的选取应从以下方面考虑：
            1.选择最能表达数据的主要特征来构建基于距离或基于概率分布度量的特征空间。
            2.选择合理的特征归一化方法，使特征向量转换后仍能保持原来数据的核心内容。
            3.选取合理的损失函数，在实验的基础上，依据损失不断调整模型的参数，使其尽可能实现类别区分。
            4.合理组合不同的损失函数，发挥每个损失函数的优点，使它们能更好地度量样本间的相似性。
            5.将数据的主要特征嵌入损失函数，提升基于特定任务的模型预测精确度。
            
**什么是优化器？**

            **优化器的概念（就是如何向后更新参数）**
              •深度学习的目标是通过不断改变网络参数，使得参数能够对输入做各种非线性变换拟合
            输出，本质上就是一个函数去寻找最优解，所以如何去更新参数是深度学习研究的重点。
             •通常将更新参数的算法称为优化器，字面理解就是通过什么算法去优化网络模型的参数。常用的优化器就是梯度下降。
             
**优化器分类**

            一、梯度下降法(Gradient）
              函数的梯度方向表示了函数值增长速度最快的方向，那么和
            它相反的方向就可以看作是函数值减少速度最快的方向。对机器
            学习模型优化的问题，当目标设定为求解目标函数最小值时，只
            要朝着梯度下降的方向前进，就能不断逼近最优值。根据用多少
            样本量来更新参数将梯度下降分为三类： BGD SGD MBGD
             （1） BGD Batch gradient descent: 每次使用整个数据集计算损失
                  后来更新参数，很显然计算会很慢，占用内存大且不能实时更新，
                  优点是能够收敛到全局最小点，对于异常数据不敏感。
             （2） SGD:Stochastic gradient descent: 随机梯度下降，每次更新度
                  随机采用一个样本计算损失来更新参数，计算比较快，占用内存
                  小，可以随时新增样本。这种方式对于样本中的异常数据敏感，
                  损失函数容易震荡。
             （3） MBGD: Mini batch gradient descent: 小批量梯度下降，将 BGD
                  和 SGD 结合在一起，每次从数据集合中选取一小批数据来计算损
                  失并更新网络参数。
-------------------------------------------------

               二、动量优化法
                    动量优化方法是在梯度下降法的基础上进行的改变，具有加
                  速梯度下降的作用。一般有标准动量优化方法 Momentum 、 NAG
                  Nesterov accelerated gradient ）动量优化方法。
                    1. Momentum
                    使用动量(Momentum) 的随机梯度下降法 ( SGD)，主要思想是
                  引入一个积攒历史梯度信息动量来加速 SGD 。动量主要解决 SGD 的
                  两个问题：一是随机梯度的方法（引入的噪声）；二是 Hessian 矩
                  阵病态问题（可以理解为 SGD 在收敛过程中和正确梯度相比来回
                  摆动比较大的问题）。
                   2. NAG
                    牛顿加速梯度（NAG, Nesterov accelerated gradient ）算法，是
                    Momentum 动量算法的变种。 Nesterov 动量梯度的计算在模型参数
                    施加当前速度之后，因此可以理解为往标准动量中添加了一个校正因子。
----------------------------------------------------------------------------

                •三、自适应学习率优化算法
                   •自适应学习率优化算法针对于机器学习模型的学习率，
                  传统的优化算法要么将学习率设置为常数要么根据训练
                  次数调节学习率。极大忽视了学习率其他变化的可能性。
                  然而，学习率对模型的性能有着显著的影响，因此需要
                  采取一些策略来想办法更新学习率，从而提高训练速度。
                    •1.AdaGrad 算法：主要优势在于不需要人为的调节学习率，
                   它可以自动调节；缺点在于，随着迭代次数增多，学习
                  率会越来越小，最终会趋近于 0 。
                    •2.RMSProp
                  算法：修改了 AdaGrad 的梯度积累为指数加权
                  的移动平均，使得其在非凸设定下效果更好。
                    •3. AdaDelta
                  算法：在模型训练的初期和中期， AdaDelta 表
                  现很好，加速效果不错，训练速度快。在模型训练的后
                  期，模型会反复地在局部最小值附近抖动。
                    •4. Adam
                  算法： Adam 通常被认为对超参数的选择相当鲁
                  棒，尽管学习率有时需要从建议的默认修改。
                  
**优化器的选择**
            
               •那种优化器最好？该选择哪种优化算法？目
             前还没能够达达成共识。具有自适应学习率
             的优化器表现的很棒，不分伯仲，没有哪种
             算法能够脱颖而出。
               •目前，最流行并且使用很高的优化器（算法）
             包括 SGD 、具有动量的 SGD 、 RMSprop 、具有
             动量的 RMSProp 、 AdaDelta 和 Adam 。在实际
            应用中，选择哪种优化器应结合具体问题；
            同时，也优化器的选择也取决于使用者对优
            化器的熟悉程度（比如参数的调节等等）。
            
**如何确定模型已经训练的能达到要求了？**
**使用过拟合的概念来解决**

![36](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a7d06d2b-c7e5-4360-9d2c-92433a58ad5f)

            **过拟合概念**
              •过拟合是指学习时选择的模型所包含的参
            数过多，以至于出现这一模型对已知数据
            预测的很好，但对未知数据预测得很差的
            现象。这种情况下模型可能只是记住了训
            练集数据，而不是学习到了数据特征。
              •具体表现就是最终模型在训练集上效果好；
            在测试集上效果差。模型泛化能力弱。
              •优化
            是指调节模型以在训练数据上得到最佳性能。
              •泛化
            是指训练好的模型在前所未见的数据
            测试集 上的性能好坏。
            
 **过拟合的原因**
            
 ![37](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a683b3d6-6e8c-4021-87b6-7fc77944d4dc)

  **解决过拟合的办法**
  
            一、数据方面
              1.从数据源头获取更多数据。
              2.根据当前数据集估计数据分布参数，使用该
            分布产生更多数据。数据增强（ Data
            Augmentation） 通过一定规则扩充数据。如
            物体在图像中的位置、姿态、尺度、整体图
            片明暗度等都不会影响分类结果。我们可以
            通过图像平移、反转、缩放、切割等手段将
            数据库成倍扩充。
              3.保留验证集。
              4.获取额外数据进行交叉验证。
           二、模型方面
             1.降低模型复杂度：
              a. 对于神经网络：减少网
            络的层数、神经元个数等均可以限制网络的拟合能力； 
              b. 对于决策树：限制树深，剪枝，限制叶节点数量； 
              c. 增大分割平面间隔。
             2.特征选择、特征降维。
             3.提前停止：及时地结束不必要的训练过程。
             4.正则化（限制权值 weight decay ）：将权值的大小作为惩罚项加入到损失函数里。
             5.增加噪声：在输入中、权值上、网络响应等加上噪声。
           三、ensemble
             1.Bagging:
            从训练集中自助采样，训练多个相互
            独立的弱学习器，通过一定结合策略形成一
            个强学习器。
             2.Boosting:
            初始化训练一个基学习器 根据表
            现调整样本分布（预测错误的样本在后续收
            到更多关注） 训练下一个基学习器 多个
            学习器加权结合。


**那如果模型还没把特征学倒位就停了，怎么办？**
**欠拟合的和原因**
![38](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/446744af-75d3-4f60-a1ae-d38fa9d61052)

              概念：欠拟合是指对训练样本的一般性质
            尚未学好。在训练集及测试集上的表现都
            不好。无法有效反应数据的特征规律，因
            此也缺少了泛化能力。
              原因：
            （1）模型复杂度过低
            （2 ）特征量过少
            
 **欠拟合的解决办法**
            
            1.增加特征项；
            2.添加多项式特征项；
            3.减小正则化系数；
            4.增加模型复杂度。
            总结：
            •训练误差指的是训练集样本与结果的误差；
            •泛化误差指的是模型预测情况与真实样本的误差；



**Deep learning based sequence model for predicting variants**

**基因组学中快速实现深度学习模型的包装器keras_dna**

            
              通过Keras_dna ，可以轻松实现深度学习模型，以便从序列中预测基因组注
            释。它可以处理输入和输出的大量文件类型。可以处理标准的生物信息文件
            格式作为输入，如 bigwig gff bed wig bedGraph（经过注释的文件） 或 fasta(原始文件) ，并返回用于模
            型训练的标准化输入。
              •使用Keras 模型 (Tensorflow 高级 API) 快速开发深度基因组应用程序，可以实现
            现有的模型，同时也促进具有单个或多个目标或输入的模型开发。
              •注释文件（如BED 、 GFF 、 GTF 、 BIGWIG 、 BEDGRAPH 或 WIG ）以及基因组文
            件（ FASTA ）很容易集成到具有多种功能（序列窗口选择、多输入 物种预测）
            以及评估（ AUROC 、 AUPRC 、相关性）的深度学习模型中。
            
            
**输入数据**
![39](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a3f501da-3cc0-401e-a6f4-3c1aff766f46)


                  **关于使用Keras DNA有3个步骤，在输入部分分为：
                  分类，回归，多种输入，首先导入数据，fasta文件，然后进行分类和注释他们**
              •Keras_dna使用基因组注释的大多数标准生物信息文件格式作为输入。 Bed gff 和 gtf 常用于分类问题，
            而 wig big wig 和 bedGraph 用于回归问题。所有这些不同的文件可以一起用于多输入问题。
              •标准输入由一系列特征和相应的标签组成。直接从 FASTA 文件中提取 DNA 序列，生成器将其作为字
            符串或单热编码输入输出到模型。它还提取相应的标签（如 BED 、 GFF 或 GTF ）或连续数据文件（如
            BIGWIG 、 BEDGRAP 或 WIG ）。为了在训练集和测试集之间实现分离，我们的生成器可以被限制为只来自特定染色体序列的生成序列。
            
            
**Generator**

              •Generator
            是一类容易调优的生成器，它处理从数据生成网络输入所需的所有数据处理。
              •multigenerator
            同时在多个数据类型和多个物种上训练模型。 Multigenerator 拥有与
            Generator 相同的功能。 不管输入的类型是单一序列还是多个序列和注释，
            ModelWrapper 都接受这些输入，通过Generator 类生成适当的用户定义批处理和标签。


**Keras_dna ModelWrapper**

![40](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/af8e942a-d730-428b-b00e-6f9631f361a5)

**相关文献**
https://pubmed.ncbi.nlm.nih.gov/33135730/

            其中Generator是用于整合raw数据的lable和序列。并转化成网络输入需要的数据.
            然后用Keras_DNA搭建神经网络，再进行训练.
            
 **以下使用方法和介绍来自于文献中的GitHub库**
 
            Keras_dna 是一种 API，可帮助快速实验将深度学习应用于基因组学。它可以快速为 keras 模型（tensorflow）提供基因组数据，
            而无需费力的文件转换或存储大量转换后的数据。它读取最常见的生物信息学文件并创建适用于 keras 模型的生成器。

              如果您需要一个库，请使用 Keras_dna：

            允许快速使用标准生物信息学数据来提供 keras 模型（现在是 tensorflow 的标准）。
            帮助格式化数据以满足模型的需要。
            促进具有基因组学数据（相关性、AUPRC、AUROC）的模型的标准评估
            
  ---------------------------------------------------------------------------
            
              keras_dna 的核心类是Generator，用于为 keras 模型提供基因组数据，并将ModelWrapperkeras 模型附加到其 keras_dna Generator。
            Generator创建与所需注释相对应的 DNA 序列批次。

               第一个例子，Generator产生对应于给定基因组功能（此处为结合位点）的 DNA 序列的实例作为正类，其他序列作为负类。
             基因组通过 fasta 文件提供，注释通过 gff 文件提供（可能是一张床），DNA 是单热编码的，我们想要靶向的基因组功能需要在列表中传递。
             
   -------------------------------------------------------------------------------
   
            **Code**
            from keras_dna import Generator

            generator = Generator(batch_size=64,
                      fasta_file='species.fa',#导入fasta文件
                      annotation_files=['annotation.gff'],#提供的注释分类文件
                      annotation_list=['binding site'])# 在哪里结合的数据.
                      
  Generator拥有很多关键字来使数据格式适应keras模型和手头的任务（预测序列在不同细胞类型中的基因组功能，在几个不同的功能之间进行分类，从两个不同的输入进行预测，标记DNA序列具有它们的基因组功能和实验范围......）

ModelWrapper是一个旨在将 keras 模型统一到其生成器的类，以简化模型的进一步使用（预测、评估）。

                  from keras_dna import ModelWrapper, Generator
                  from tensorflow.keras.models import Sequential()

                  generator = Generator(batch_size=64,
                                    fasta_file='species.fa',
                                    annotation_files=['annotation.bw'],
                                    window=100)
                      
                  model = Sequential()
                  ### the model need to be compiled
                  model.compile(loss='mse', optimizer='adam')
 
                  wrapper = ModelWrapper(model=model,
                                    generator_train=generator)

在构建完模型后可以根据需要使用功能函数来进行模型的训练，评估，预测，已经包装.

                  训练模型.train()

                  wrapper.train(epochs=10)#epochs= （）这是让你确定进行训练的次数
                  在染色体上评估模型.evaluate()

                  wrapper.evaluate(incl_chromosomes=['chr1'])#评估loss值，在验证集上的测试
                  预测染色体.predict()

                  wrapper.predict(incl_chromosomes=['chr1'], chrom_size='species.chrom.sizes')#预测loss值
                  将包装器保存在 hdf5 中.save()#当评估数据和预测数据都符合标准时候，就可以进行储存.

                  wrapper.save(path='./path/to/wrapper', save_model=True)
                  
                  

# 如何very clear-eye about what to tune in order to try to achive one effect.(这个过程叫做正交（orthogonalization）)

这个Orthogonalization就像是老式电视机上的knobs，可以通过他们来控制电视画面位于中间，
假如她有4个knobs，就1个是调节高度，一个是调节宽度，一个是调节横纵比，一个是调节翻转.
这4给knobs共同工作，使得画面位于正中，so in this context,就是设计师
保证每一个knob只能对一个参数进行调节Orthogonalization.


![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d05aa96c-2f3a-4df8-926e-06efdecd981a)

**例如在车辆中**
车是由3个装置控制的，1. steering（方向盘）
                   2. acceleration（加速度，以及油门）（controls seep）
                   3. braking（刹车，制动）（controls seep）
it makes it relatively interpretable. 他们让指令相对可以解读.


## orthogonalization 和 machine learnin的关系

For a supervised learning system to do well, you usully need to tune
the knobs of your system to make sure that four things hold true.

1.  you usually have to make sure that you're at least doing well on the training set.
  (So performance on the traing set needs to pass some acceptability assessment)
2.   **Fit dev set well on cost function**
3.   **Fit test set well on cost function**
4.   **performs well in real world**

如果我们的算法对dev set的拟合结果不太好，但是在training set上效果好，就需要另外 a set of Knobs了.
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/1f48826b-7e0b-4a7c-91c9-509cde2e5cc3)

# set up your goal
## single number evaluation metric(设置一个单一化评估指标)
这个signle number evalution mertric 可以有效的提升iteration 速度.

![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/0358c148-f0e9-495a-8e27-09c1323aa43d)

在开始一个机器学习项目前，先设置一个单一化评估指标.

例如图中，我们训练得到了一个binary的问题，就平常来说判断A,B两个模型哪个的更出色.

我们是需要根据Precision 和 Recall。 但如果分开来看是难以兼顾的。

我们可以直接找一个新指标，它兼顾了准确率和找回率.

F1 score 可以认为是，"Average" of P and R.

Formally, the F1 score is defined by this formula. : 2/(1/p + 1/r)

通常在，很多进行机器学习的团队都有一个Dev set，和有一个Single Number Evaluation Metic 来帮助快速判断哪个分类器更好.


## Satisficing and Optimizing metric



例如我得到了3个classification 的model，其中有两个指标，Accuracy 和 Running time.

这时候我想要评估它的整体性能就可以加入一个 Overall evaluation metric（整体评价指标）

在这个时候，我们就可以设定一个分类器（classifier）在确保运行时间的前提下提供最大准确率.

例如： Maximize accuracy 的同时 Runing time 小于 100ms.

![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/56562fc5-fa58-488b-933a-87ec3ad0528c)


**所以more generally 如果有N个metrics是被关心的**

有时选择一个进行优化是合理的，你想要它表现的足够好.

而N - 1 的metrics， 我们只需要他们 Satisficing(满足) 一些阈值（threshold），只要到了threshold就不用管他们Optimize了.

如果我们正在构建一个检测唤醒词（wake words）的系统， 也叫触发词（trigger words）.

它指的是，一些声控设备。 例如用 Hey Siri 来唤醒Apple设备.

你可能会关心这个唤醒词的准确性（accuracy） 就当trigger words被喊出后，有多大的可能唤醒设备.

也许也会关心false positive （假阳性）的次数。也就是别人叫了trigger words但是没有响应的次数.

那构建single nuber evaluation metric的时候就可以限定，一天最多出现1词false positive做为threshold，
而Maximu accuracy是优化指标.

## 如何分布Train/dev/test sets.

How to set up the dev/test.

**选择distribution of the Train/dev/test**

是需要根据 dev set + metric 定制的一个目标，然后就可以尝试不同的方法达到这个目标.

为了使得数据没有偏向性，我们需要把数据随机打乱，然后再分成dev和test set.


**Guideline**

**其中的dev和test set应该有same distribution**

Choose a dev set and test set to reflect data you expect to get in the future and consider important to do well on.

## Size of dev and test sets

in the old way of splitting data.

In the early machine learning, this is very reasonable. 

70% traning set, 30% testing set.

or the 60% traning, 20% develop set, 20% test set.

如果样本量足够大，例如达到10000000，那我们就可以选择98% as the traning set, 1% as the dev set, 1% as the test set.
![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/3bad0d14-a7c8-49dd-a6f4-356269c8a7eb)

**Size of test set**

Set your test set to be big enough to give high confidence in the overall performance of your system.

这就意味着，也许test set并不需要上百万给样例. 也许1W个样本就能提供足够的置信度来评估性能，或10w给.

## When to change dev/test sets and metrics

好办法其实是加入error 分数的筛选。

Error： 1/ m_dev * (i=1 到M_dev数量的连加) y（预测）≠ y（实际的值）
但是这时，这个并没有把我们需要区分的色情图片错误和非猫咪错误区分开来.
我们可以加入一个进行判断了W参数来解决这个问题.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/b4e1a34f-dfd7-4452-a0d4-aebd89c171b1)


# 关于机器学习目标设定的步骤

**可以更快的对算法A还是算法B更好做出决定**

1. 设定评估功能的单一化指标（first step： play target）（由一个knob控制）
2. 考虑如何在这个指标上获得更好的性能（第二步，研究如何打中target）（由另一个knob控制）
3. 可以通过在Cost function中引入常数（）的方式增加权重，来达到获得更好性能的目的.

![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d475c1fa-5fa2-4261-a8ca-d9d0a4308140)

例子
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/9682c7cc-c052-4f7f-8caa-baba1bcc7cc2)


## Comparing to human-level Performance

### Why human-level performance

1. 在某些领域中，机器学习处理问题的程度能和人比较了.

**特别是在训练机器学习算法，如果以时间作为横轴可以看到，一段时间之后，机器学习算法的准确读就高于人类了**

但是都会低于贝叶斯最优误差线. 当你准确率低于人类的时候，还有工具来进行改正，如果accuracy高于人类了，
就很少有工具能进行提升准确率了.

![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/000dbbe5-c124-4cfe-8b81-2fb8ff1400e2)

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/94e82cf1-a31e-40d6-90b0-9c3490ee82ea)

### 可避免的bias

**将Human类误差和Training error之间的差值看作是Avoidable Bias**
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/e10710c2-7c1e-4da3-bdf4-048cc7053668)


例子： 在执行猫识别任务中，如果训练集（8%）的误差度和人（1%）都有比较大差距（bias高了，训练set不拟合）
1. 训练更长时间，用更大的神经网络（更多层）

例子例子： 在执行猫识别任务中，如果训练集（8%）的误差度和人（7.5%），而Dev set中error（10%）.这个时候，就是模型在Dev set中不拟合了，这个时候就是过拟合训练集（high Variance）.

1. 进行Regulazition 减小Variance

**所以我们可以将人类准确度看成是贝叶斯最优误差的代理变量，或者是估计值**

上面两个例子表明其实，是否overfitting或者是underfitting都是取决于人类误差的比较的.

所以我们需要进行判断，是Avoidable bias大？还是Variance误差大


### Understanding human-level performance

more precisely define the human-level performance.

**Human-level error as a proxy for Bayes error**

例子中，对于这张骨科X-ray image不同的人或者团队会有不同的准确度.
其实选择哪一个作为human-level performance是基于你的模型的适用场景的.
最佳的肯定是准确率最高的0.5%，但是如果只是进行发表，或者证明有效性，1%超过一个普通放射科医生就可以了.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a159975f-6249-437e-9036-b395c4f3cd9b)

**选取不同level作为human-level时候**

Human-level error 和 Training error之间的误差就是Avoidable bias

这个Training error和Dev error之间的误差就是表现了Variance的值的变化.

### Surpassing Human-level Performance

一旦它的training error 解决0.5% 就很少有工具能够提升了，但是能通过Avoidable bias来找到Optimization.

**Problems where ML significantly surpasses human-level performance**
例如：
- Online advertising （算法性能远超人类）
- Product recommendations （产品推荐）
- Logistics（predicting transit time） (预测物流时间)
- Loan approvals （贷款许可）

**这些都不是Nature Perception（自然感知）问题**
不是计算机视觉等。

而人类在Perception方面更有优势，所以导致Algorithm在这方面很难超过人类.

**当然以上所有，都是基于我们的团队获得了大量的数据的情况**

### Improving your model performance

1. 很好的fitting training set（可以得到较小的Avoidable bias）
2. The training set performance generalizes pretty well to the dev/test set(the variance is not too bad)

**下图展示了如何降低Avoidable bias以及Variance 值的方法**
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/9132bb60-b4ef-491a-97ce-ccf38dc0a8a1)

# Carrying out error analysis

**如果我们想让我们的algorithm达到human-level，我们可以手动地检查算法中的错误**

This process called **error analysis**

Look at dev examples to evaluate ideas

如图下例子中所示，这是一个识别cat的算法，它对猫的识别率是90%，而那10%error 是下面的dog图.

队友会建议  ：如何使算法更好，特别是在识别狗的时候。

所以就会出现侧重点：collect more dog pictures. 为了让这个算法不再把狗判定成猫.

![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c0f9e1b1-10af-4e3d-bc05-1d97b5f19bca)


关于 **Error analysis**：
1. Get~100 mislabeled dev set examples.
2. Count up how many are dogs.(现在图中的结果是，100张分错的样本中有5%是狗的图片)
   这意味着 就算解决了狗识别的问题，也没有提升多少准确率.
3. 如果计算出来之后，发现100张分错的样本中，有50%是狗，那就值得努力去提升准确率.
4. 所以一般进行Error analysis可以绘制一个excel图.这将帮助你遍历手动查看图像集，
   而这个Colum是进行检测的问题：1. Dog， 2. Great Cats， 3. Blurry images问题，4. 再留一个列来写conmments.
   
![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/4d6c60c3-ea4b-4045-a0da-bdb6209c1a6c)

这个图将告诉你，处理哪一个部分的问题，你的Model可以得到一个更准确的结果.

如果有两个问题都占有较大的error比重，就可以分成两队，一队改善Great cats一队改善Blurry.

总结： 要进行error analysis 你应该找到一套在验证集中被错误预测了label的样本，并按照 look for false positive and false negative. 并计算在不同类别中mislabeled的数量。

## cleaning Up Incorrectly Labeled Data

In the Supervisor learning case, the Data comes from input X and output label Y.

而在Model预测后输出的X和Ylable对不上时候，就产生了 Mislabeled examples.

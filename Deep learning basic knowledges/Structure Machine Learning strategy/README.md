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




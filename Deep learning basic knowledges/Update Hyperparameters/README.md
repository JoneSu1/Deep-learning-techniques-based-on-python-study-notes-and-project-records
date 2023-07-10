# 降低overfitting的方法

- 1.数据扩充（Data Augmentation）：通过对训练数据进行一系列的变换和增强操作，生成额外的训练样本。这可以帮助模型更好地泛化并减少过拟合。

- 2.正则化（Regularization）：通过在损失函数中添加正则化项，限制模型的复杂度。常见的正则化方法包括L1正则化和L2正则化。

- 3.早停（Early Stopping）：在训练过程中监控模型在验证集上的性能，当性能不再提升时停止训练，以避免过拟合。

- 4.Dropout：在训练过程中随机关闭一部分神经元，以防止网络对特定神经元过于依赖，增强模型的泛化能力。

- 5.参数调节（Parameter Tuning）：调整模型的超参数，如学习率、正则化参数等，以找到最佳的模型配置。

- 6.使用更多的训练数据：增加训练样本的数量可以降低模型对于特定样本的敏感性，有助于减少过拟合。

- 7.简化模型结构：减少模型的复杂度，例如减少隐藏层的数量或神经元的数量，可以降低过拟合的风险。

- 8.集成方法（Ensemble Methods）：通过结合多个模型的预测结果，如投票、平均等方式，可以降低模型的方差，提高泛化能力。

## 常用的正则化方法包括：

- L1 正则化（L1 Regularization）：在损失函数中添加 L1 范数项，即参数的绝对值之和。L1 正则化可以使得部分参数变为零，从而实现特征选择和稀疏性。

- L2 正则化（L2 Regularization）：在损失函数中添加 L2 范数项，即参数的平方和。L2 正则化可以使参数的值趋向于较小的值，避免过拟合问题。

- Elastic Net 正则化：是 L1 正则化和 L2 正则化的结合，同时引入了 L1 和 L2 范数项。Elastic Net 正则化可以在特征选择的同时保持一些相关特征的权重。

- Dropout：在训练过程中随机关闭一部分神经元，以防止网络对特定神经元过于依赖，增强模型的泛化能力。

- 数据扩充（Data Augmentation）：通过对训练数据进行一系列的变换和增强操作，生成额外的训练样本，有助于降低过拟合风险。

# Normalization

在神经网络中使用归一化（Normalization）的主要目的是将输入数据转换为具有相似尺度和范围的统一分布，以提高模型的训练效果和性能。以下是一些常见的原因：

**梯度消失/梯度爆炸（vanishing and exploding gradient）问题**：当输入数据的值具有较大差异时，激活函数的导数可能变得非常小或非常大，导致梯度消失或梯度爆炸的问题。通过归一化输入数据，可以将输入值限制在合理的范围内，有助于避免这些问题的发生。

**The vanishing and explding gradient means:** The derivaty and sloop of loss function becomes very huge or small.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/214c1287-8995-4da1-a7ea-81ff6f5155ed)

我们可以看到如果，W值被跟新到1.5的化，W（L）将会越来越大，A激活函数的值也会指数增加，如果W = 0.5,则激活函数A会指数降低. 所以如果神经网络层数过多，模型过大，对预测的结果是有显著影像的.

**提高收敛速度**：归一化可以使得输入数据的分布更加接近标准正态分布或均匀分布，这有助于加快模型的收敛速度。具有相似尺度和范围的数据可以更容易地在梯度下降过程中进行优化。

**改善模型的泛化能力**：通过归一化输入数据，可以减少模型对特定特征的依赖程度，使得模型更具有泛化能力。这有助于防止模型过拟合训练数据，并提高在新样本上的性能。

常见的归一化方法包括将数据缩放到特定范围（例如[0, 1]或[-1, 1]），或者使用标准化方法将数据转换为均值为0，标准差为1的分布。

总之，归一化是一种常用的预处理步骤，有助于改善神经网络模型的训练效果、收敛速度和泛化能力。具体的归一化方法和参数选择应根据具体问题和数据集的特点进行调整。

## 而一种能有效帮助我们减小这种可能，并帮助我们更好的初始化（initialization）parameters的方法（Weight initalization for deep nueral network）

**先从single神经网络开始**
**这展示的是当Relu作为激活函数时候**
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/f208253d-4708-4d0e-8092-c19307edfc05)

其中的n表示input的feature数量。 np.squrt()是平方根函数. 而下面是n[l-1]是因为在使用ReLU作为激活函数时，使用归一化时通常选择对前一层的激活值（n[l-1]）进行归一化，而不是对当前层的激活值进行归一化。
这是因为ReLU激活函数具有非线性的特性，其输出对于负值输入为0，对于正值输入为线性增长。由于ReLU的非线性部分只对正值起作用，因此对负值进行归一化可能会导致信息的丢失。.

**这里展示的是当tanh作为激活函数的时候**
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/e05d9ed6-e4c2-4a6a-8ef8-06d2e82fa0ef)


##  Gradient Checking
**当神经网络种完成了backward propagation之后要进行gradient descent，如何确定gradient descent倒位了,（Numerical Approximation of gradient）梯度数值近似.**
### 1. check the derivative computation
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a7db0089-2324-4dd7-9b9c-68f817389594)
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/6f549016-c4f7-48ab-9f9d-052df6fda324)

当原公式中计算出的derivative是接近g()的值的时候，那就意味着g（）这个跟新过的是合适的. 我们可以看到0.0001就是我们说的actrully exactly approximation error.

## 在检查完了derivative之后，就可以来检查gradient

- first step: Take the all of parameters(W,b) to reshape into a big verctor 0.
- second step： 将所有参数的导数 to reshape into a another big verctor d0.
 ![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/6502b0c2-9dc7-4efb-b1d8-89018af77ae4)


**关于代码实现**

首先我们需要把每一个parameter给提出来.
所以我们会用到for loop中对每一个参数都进行处理的命令
for i each in （）
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/89f26ef2-b119-44bb-bfe7-188637377ac6)

如果这个gradient checking的结果是10（-7）次方那就对了，如果是-5次方就需要检查了.

### 关于实现gradient checking的一些提示

- 不要在训练模型的时候去gradient checking，只在debug的时候使用
- If a algorithm failed in grad checking, look at components to try to identify bugs.
  (主要就是检查d0（approximation）中哪些i导致的和d0的差距过大)
  
- 记得regularization （都是处理过拟合的）
- don't work with dropout (用dropout处理过net之后，J（cost）的值不容易计算)
- 要考虑有可能是因为W，b随着iteration而变大了，精准的时候可能是W，b接近0的时候.
- 

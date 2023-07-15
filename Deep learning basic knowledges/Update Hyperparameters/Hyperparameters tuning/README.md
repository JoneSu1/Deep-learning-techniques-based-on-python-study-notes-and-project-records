# Hyperparameters Tuning
## Tuning process

**如何为这些参数设置一个合适的值呢？**

比如： **α**：learning rate，大多数时候，这是最重要的参数
      **β**： 如果是使用了动量算法（Momentum），就会有动量参数β，一般0.9是合适的
      β1，β2，epsilon： 在Adam算法中，第一个是Exponential weight means的参数，第二个是Exponential weight means of sqare的参数. 
      其中epsilon是属于经历完所有的mini-batch的gradient descent的数量.
      layers：有几个hidden layers，
      **hidden unites**：每层中有几个神经元
      learning rate decay： 很多α
      **mini-batch size**： 设置随次数增加的seed数，每次都随机分配，根据设定的mini——batch size，一般按照2的指数来选择，64是常用的.
      
      加粗了的hyper参数是比较重要的，在深度学习中.

      **如果我们有多个参数，我们可以画出square来进行25次的随机取值，然后分别执行，在选出效果最好的**

      ![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/880e3172-fcbf-4942-8476-e211a04afed5)

**也可以进行区域定位选出**

比如我们发现了一个参数可能不错，就在它周边选则更多的参数进行测试.
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/38e59d9c-7aa0-4e70-adbf-892b6dc87cee)

## Using an appropriate scale to pick hyperparameters

**使用适当的比例来选择超参数**

更重要的是选择合适的尺度（scale）。

**Picking hyperparameters at radom**

首先，假设 我选择在hidden unite的数是n[l] 在第一层layer中。
我们认为 50到100直接是一个好的选择。

我们就可以在这个范围中随机选择一些数，来测试.

如果需要确定的是layer数量（L）：2-4之间。
就进行随机抽样，234来进行测试，来看结果.
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/8cf34fde-17ba-429d-a275-8ddc36c6c142)

**但是这种方法不适用于所有的Hyperparameters**
比如，我要确定learning_rate（α），它可能的区间就是1到0.0001. 如果我进行随机采样，范围太大.

这个时候我们就可以考虑适用log（scale）进行，而不是用linear scale了.

我们可以设对数 r = -4 * np.random.randn()，这就意味着，r是 -4 到 0随机取值
然后 取α的值， α = np.power(α，r) 。而α将会取， α的0次方到α的-4次方的值.
![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/145b6221-0ac3-411d-bbc5-dc19462a7300)

**关于β的取值**
我们如果进行Momentum算法，就需要定义β的值.

而关于β我们一般是（0.9，0.999）之间取值。
如果直接进行随机采样效果不好。
我们可以取用1-β处理，就会变成（0.1，0.001）就成了10的-1次方到10的-3次方的范围了，就可以用log scale来取值了.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d208da92-e5e0-4eac-9586-b63aef9f5ed9)

## 关于模型训练的方法
- **方法分为两种，一种是专注于一个模型，并用不同的方法进行改进并看cost图** pandas
- **第二种** 同时并行多个模型进行训练，然后找出最合适的. Caviar 方式

- 而如何选择用哪种，这主要是取决于你的算力，如果算力支持高，就可以直接用Caviar的方式进行
![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/46562179-e803-4073-9366-78bf3c9eef35)

  # 我们可以用 should be: X = X/\sigmaX=X/σ 来帮我们正则化batch，方便我们更好的找到参数.
## Batch Normalization

可以让我们的Hyperparameters 更具有鲁棒性.

**Recall**

之前我们就知道，用每一个数除以算数平方根的值，来达到数据降维。这将帮助我们加快学习.

而如果我们对每一个hidden layer的Z(L)值都进行归一化，就可以到达对所有NN值降维的目的（就是batch normalization）

而我们不想让不同分布的所有hidden layer的Z值都被相同处理所以加上了β和γ（gamma）.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c7de6bac-e6f9-42b2-8883-3941fe12f7ee)

**如何应用batch Normalization，在deep neural network 中**

我们认为每一个圆圈都代表着两步计算.

如果我们不使用batch Normalization，那决定下一层Z值的就是上一层的W,b.

如果我们使用BN，则下一层是由β（l-1）和gamma（l-1）决定Z值.

我们可以用dBeta L（L层Beta的导数）来跟新.
除了用常规的gradient descent取跟新，也可以用Momentum，Adam的方式进行更新.
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/d264e38d-ee41-4cb7-b063-dc1e7c9c69a9)

**同样我们可以使用mini-batches的方法来跟新Z值**
![5](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/7d5b566c-1894-4fa8-ac8c-5b7648156496)

总结：
![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/edd6f0f5-f4d4-4003-8114-6abe19dad195)

**为什么使用batch-norm**
因为，当每次迭代时候，每一个hidden layer的parameters都在变化，它的distribution并不统一，所以对Z进行norm会有利于训练.


## 如何测试经过了Batch Norm的值

**使用到的公式**
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a0dc90d1-ffb2-4796-a855-d62ac515f68f)

**我们通常用指数平均数来评估，Exponential weight average（根据mini batch来计算）**
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/4527a976-c20a-4559-abd8-78502d4ed2be)

# 学习使用workframe： TensorFLow
这将更加高效的帮我构建模型.

**以下是常见的work frame**
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/4a8c7f6d-31b6-4dda-b500-5b678132caf6)


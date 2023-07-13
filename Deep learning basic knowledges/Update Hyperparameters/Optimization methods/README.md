# 优化参数iteration的方法
 1. **minibatch-gradient descent**
 2. **Exponentially weighted averages(指数加权平均)**
 3. **值得了解的技术bias correction in Exponentially Weight Averages**
 4. **Gradient Descent with Momentum(动量梯度下降)**(比上面的标准Optimization算法更快，计算加权平均值然后跟新权重)
 5. 结合了Momentum和RMSprop（root mean square propagation，均根传递）的最优秀算法**Adam**
 6. **Learning rate decay**
**当数据较大时候，一个好的Opimization将会缩短很多时间**

## Mini-batch gradient descent是一种用于训练神经网络的优化算法。
在mini-batch梯度下降中，训练数据被分成小批量（mini-batch），每个小批量包含多个训练样本。然后，针对每个小批量，计算梯度并更新模型的参数。

相比于批量梯度下降（batch gradient descent），即一次使用所有训练样本计算梯度和更新参数，mini-batch梯度下降的优势在于更高的计算效率。通过使用小批量，可以并行地处理多个样本，充分利用硬件资源（如GPU）加速计算。

通常，mini-batch梯度下降具有以下优点：

更快的收敛速度：与批量梯度下降相比，每次更新只基于一部分训练数据，可以更频繁地更新参数，加快收敛速度。
更好的泛化能力：小批量梯度下降可以避免陷入局部最优解，并更好地泛化到新数据。
内存效率：相比于一次加载所有训练数据，使用小批量可以降低内存要求，尤其在处理大规模数据集时更加有效。
但是，与批量梯度下降相比，mini-batch梯度下降也存在一些缺点：

需要调节学习率：由于每个小批量只是训练数据的子集，梯度估计可能会有噪声，因此需要适当调整学习率。
可能陷入局部最优解：由于每次更新只基于一部分数据，可能会导致模型陷入局部最优解而无法达到全局最优解。
总之，mini-batch梯度下降是一种折中方案，结合了批量梯度下降和随机梯度下降的优点，是训练神经网络时常用的优化算法。

**在之前我们知道了使用Vectorization可以帮我我们省下使用for loop的时间**

Vectorization allows we to efficiently computer on m examples.

如果m过大，也同样会使得traning rate降低，所以，我们可以将traning set 拆分成更小的traning set（little baby training sets）
叫做：(mini batch), 如果说m = 50000，我们就设置mini-batch of 1000 each. 而每次训练是一个batch一个batch的训练，
不是全部一起.
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/82d107d3-1ff1-4c63-befd-8039954ea985)

Mini-batch t : X{t},Y{t}
```
x = (n,m)
x{1}=mini-batch 1 = x1,x2.....x1000 = (nx,1000)
x{2}=mini-batch 2 = x1001,x1002....x2000= (nx,1000)
.....
x{50} = mini-batch 50 = ..........
```
```
Y = (1,m)
Y{1}=mini-batch 1 = x1,x2.....x1000= (nx,1000)
Y{2}=mini-batch 2 = x1001,x1002....x2000= (nx,1000)
```
1 step ： 用（X{t}，Y{t}）做gradient descent
全部样本有1000个
```python
for t =1,....1000

forward prop on X{t}
Z[1] = W[1]*X{t} + b[1]
A[1] = g[1]*(Z[1])
...
A[L] = g[L]*(Z[L])

#这是用vectorization处理1000给example

Computer_cost J = 1/1000 * (A,Y) + （入/2*1000）||W[L]||2

Backward to comput gradient cost J{t} (X{t},Y{t})

W[l] = W[l] - learning_rate * dW[l]
b[l] = b[l] - learning_rate * db[l]
#以上的代码用和普通gradient descent的方式过了一遍，但对象只是1000个samples。
#这个过程叫做遍历（epoch）
```
**如果我们用if或者while loop就可以多次实现epoch**

在Mini-batch gradient descent中每一次的iterator都是对X{t}Y{t}的处理，所以会看到Cost迭代图是有noise的并不是直线下降.
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/fad36c1f-8667-4d42-b93c-9ceca632d6ba)

### Choosing your mini-batch size
extreme case 
- 1.If mini-batch size = m: Batch gradient descent. (X{t},Y{t}) = (X,Y)
- 2. If mini-batch size = 1: stochastic(随机) gradient descent: Every example is  a mini-batch
- 3. Batch gradient descent 的cost曲线是很少noise的（蓝色线）， stochastic gradient descent的cost曲线是多noise的（紫色线）
- 4. mini-batch 则是处于batch gra和stochastic gra的中间（绿色线）
- 5. If small training set: 就用batch gradient descent(小于2000 examples).
- 6. 一般选择64到512作为mini-batch size，再大就是1024（2的12次方）（多尝试2的幂数来尝试寻找最佳cost）.
  

  ![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/3f69799a-edd1-4acb-96b3-23e19a8197ff)
  

  ## 第二种Optimization：Expomentially weight averages

   如果想要找出temperature图中数据的曲线，并不让noise影响
```
  将V0设为0
  则V1 = 0.9vo + 0.1*当天温度
  V2 = 0.9Vo + 0.1*第二天温度
  Vt = 0.9Vo + 0.1*第天温度
```
  然后在图中绘出曲线，这就是每日温度的Expomentially weight averages

  ![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c9ab6e98-ec4c-4e18-a753-e219c61f0981)

**关于0.9这个值的选择**

1. 如果选择这个值更接近1，取0.98，这时候：1/（1-0.98） = 50（相当于粗略算了前50天的温度）.
2. 这个值选的大，就会曲线更平滑，但是会发生曲线右移动.（因为数值大计算时候产生了数值延迟，适应的慢）
3. 这个值选小的（0.5），它算出来就是相当于2天的估计。噪声多，更快适应温度的变化.
4. 所以根据下图推出的，平均值是约等于（1/（1-设定的数））
![6](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/de8a7d54-dd29-4e2d-8f9b-492e37b1371a)

**关于如何使用**
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/1301df83-2fd5-4da0-bc6a-d85240c9603e)



## Bias correction in exponentially weight averages

Bias correction 会让我们得出的averages 更accurate.

下图中：gree——line： β = 0.98. purple——line： β

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/bea1bcf4-92f7-43ab-acbc-f2fc7e2d81c0)



## 对这个技术的实践，Momentum.

因为使用gradient descent的时候，它的学习轨迹是折线的，如果iteration太大会超出范围。

**而Momentum算法可以减少前往最小值路上的震荡，因为他是取了两个偏差的平均值，能更接近直线的得到最小值.**

**可以看成在求一个weight数据分布为碗的数据时候，dW，db是小球从碗边落下的加速度，把β看成摩擦力（因为β小于1）**
![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/1951e368-e3ed-4880-af84-e058fc5d9279)
```
 Momentum
  on iteration t:
  computer dW,db on current mini-bath
  VdW = βVdW + (1 - β)*dW （就是刚刚的expomential weight averages）
  vdb = βVdb + （1- β）*db
```
  **新的权重**
  如之前探索的，expomentaily weight averages中，最合适的β值是0.9
```
  W = W - α*VdW
  b = b - α*Vdb 
```
**完整的公式**

记得把Vdw = 0， Vdb = 0
![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c21575fc-8dbc-4c6a-a1f7-846f87e4a432)

## 一种新的Optimization algorithm（RMSprop（Root Mean Square prop(均方根传递)））

**我们是通过W和b来影响这个迭代的方向的，如果iteration rate太大就会出现超偏，而如何把这个迭代速率最大化？**

 就是Root mean square prop algorithm 做的事.

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/52d40dc3-3f92-4eaf-b3d1-ff9a4e9381c4)

**蓝色的是未经Optimization（root mean square propagation）的，绿色是收敛了b的**

**根据图和公式，我们希望dW^2是小的，W就是大的，这就会往直线走，db^2是大的，那么b就是小的，那像上的偏差就少.**

On iteration t:

       compute dW,db on current mini-bath
       SdW = β*SdW + （1-β）* dW^2
       Sdb = β* Sdb + （1-β）*db^2
       #然后用以下方法跟新parameters
       W：= W - α * （dW/根号下SdW ）
       b：= b - α * （dW/根号下Sdb ）

## 结合Momentum 和 RMSprop的最流行算法，Adam

   VdW = 0, SdW = 0, Vdb = 0,, Sdb = 0
   On iteration t:
      compute dW,db using current mini-bath
      #V是执行Momentum
      VdW = β（1）*VdW + （1-β（1））*dW
      Vab = β（1）Vdb + （1-β（1））*db
      #S是执行RMSprop
      SdW = β（2）*SdW + （1—β（2））*dw^2
      Sdb = β（2）*Sdb + （1—β（2））*db^2）

      #bias correction
      VdW = Vdw/(1-β（1）^t)
      Vdb = Vdb/（1-β（1）^t）
      Sdw = Sdw/（1-β（2）^t）
      Sdb = Sdb/（1-β（2）^t）

      #gradient descent
      W = W - α*（Vdw/(根号下SdW+ Epsion)）
      b = b - α*（Vdw/(根号下SdW+ Epsion)）

### Hyperparameters choice:

     α： needs to be tune
     β1：0.9                                    （dw^2）
     β2：0.999                          （dw^2）
     E(Epsilon):   10^-8

  Adam: adaptive moment estimation(自适应估计)


## learning rate decay(学习效率衰减)

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/ffeee178-40cd-43a5-ad81-5cee385189b8)


有时学习率过高了，超过了minimize的点，那时就需要进行learning rate decay.

学习率衰减的目的是在训练过程中逐渐降低学习率，使模型在接近最优解时更加稳定。这样可以提高模型的收敛速度，并帮助模型更好地泛化到未见过的数据。选择适当的学习率衰减方法和衰减率是优化神经网络训练的重要考虑因素。

在深度学习中，学习率（learning rate）是一个非常重要的超参数，它决定了模型参数在每次迭代更新时的步长大小。合适的学习率可以加快模型的收敛速度，提高性能，而不合适的学习率可能导致模型无法收敛或者陷入局部最小值。

Learning rate decay（学习率衰减）是一种在训练过程中逐渐降低学习率的技术。通常情况下，一开始使用较大的学习率，然后随着训练的进行逐渐减小学习率。这种技术有以下几个原因和好处：

收敛性：较大的学习率可能导致训练过程中参数更新过大，无法找到合适的最优解。通过逐渐减小学习率，可以使参数更新更加稳定，有助于模型收敛到更好的解。

防止震荡：有时候学习率过大会导致参数在最优解附近来回震荡，而无法收敛。通过衰减学习率，可以减少震荡的可能性，使训练过程更加平稳。

稳定性：衰减学习率可以使模型在训练后期更加稳定。在接近最优解时，模型可能只需要微小的参数更新，而较大的学习率可能会导致模型跳过最优解。通过衰减学习率，可以使模型在接近最优解时更加谨慎地更新参数。

更好的泛化性能：学习率衰减也可以帮助模型具有更好的泛化性能。通过逐渐减小学习率，模型可以在训练过程中逐渐细化参数调整，从而更好地适应数据集的特征，减少过拟合的风险。

**常见的Learning rate decay**

![3](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/a7f39862-5e75-4141-8804-a0b9e746eac0)


定期衰减（Step Decay）：在训练的特定时间点或特定的训练轮数之后，将学习率乘以一个衰减因子。例如，每隔一定的训练轮数，将学习率减小为原来的一半。

指数衰减（Exponential Decay）：以指数函数的形式逐渐降低学习率。学习率的衰减速度取决于指数函数中的衰减率参数。


自适应学习率方法（Adaptive Learning Rate）：根据模型训练的情况动态地调整学习率。例如，AdaGrad、RMSprop 和 Adam 等优化算法就会根据梯度的变化情况自适应地调整学习率。

### how to implement learning rate decay

**主要是根据epoch的增长来使得学习率α下降，所以公式中是一个分式 * 上一次α的形式**

![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/4b5efef1-541e-4c72-b9e5-1d711739f3f2)


**Dr.Andew书写错误的订正**

![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/fd63296f-c56a-4667-99cc-b408f19504a3)

## local optima(局部优化) 

最优解，gradient descent的大多数点是位于边上的鞍点（Saddle point）这些点就是grad为0的点。而不是所有点都是0.

而大多数时候我们遇到的最优解都是像右图这样的，不同曲线交点得到local 解，而不是局部最优解.
而这个解看起来像马鞍，而那个点刚好导数是0， 所以才会叫鞍点.

![4](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/b269d80d-77d2-4207-8565-7ad447d3fc76)


**Problem of plateaus(停滞区)**

可以使用(Momentum) algorithm 和RMSprop 以及 Adam算法解决进入palteaus的问题.

他们能快速的学习并通过这个区域得到优解.
 ![6](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/227f8a47-0a84-4b0a-8a1a-0d1f3c1c6645)



plateaus 是指derivative长时间接近于0的一段区域.

如图，从那个点开始进行gradient descent，由于grad为0或者接近0，the surface is quite flat.
所以可能会花费很长的时间，取缓慢的在plateaus中找到那个点.


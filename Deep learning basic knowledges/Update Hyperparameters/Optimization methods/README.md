# 优化参数iteration的方法
 1. **minibatch-gradient descent**
 2. **Exponentially weighted averages(指数加权平均)**
 3. 
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
...
  将V0设为0
  则V1 = 0.9vo + 0.1*当天温度
  V2 = 0.9Vo + 0.1*第二天温度
  Vt = 0.9Vo + 0.1*第天温度
...
  然后在图中绘出曲线，这就是每日温度的Expomentially weight averages

  ![2](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/c9ab6e98-ec4c-4e18-a753-e219c61f0981)

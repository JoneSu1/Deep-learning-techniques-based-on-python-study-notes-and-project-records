## 理解sequence modle是如何工作的

它的输入有两个部分：X的输入，和Y的输出.

**假设我们要从一段文字中识别出哪些词是人名**

最简单的输出方式就是，对应X目标的块，我们定位Y的输出为1，不是的就是为0.

而为了更加清楚的明白他们的排序，我们可以对每一个单词都看作是X(1)....X(N)然后进行对应
那同时Y要Y(1)....
然后此时我们就已经确定了序列的block了，就可以用Tx=9来表示我确定出了9个block模块=序列长度.


![1](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/9aebca0a-eb78-4ecb-aa75-feebf5524ba3)


## 这时候引入Representing words 自然语言处理

我们需要提前准备好Vocabulary dictionary（将可能用到的单词放到一起）




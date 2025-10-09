# 摘要大致分析
transformer使用编码器解码器架构

编码器 首先将输入的词转化为词向量（长度为n） 然后进行处理 编码器可以看到整个句子

解码器 无法看到整个句子 将解码器的输出做为输入 进行处理 （一个词一个词生成） 同时 在生成yt的时候 y1~yt-1都会作为输入 从而共同生成yt 也叫自回归

<img width="1065" height="890" alt="image" src="https://github.com/user-attachments/assets/8fe29afd-64c8-415f-860c-dab30408795e" />
然后是典中典的 transformer的图 讲滴非常清楚

讲一下个人的理解 就是

首先把句子转化为一个个token 然后再进一步变换为token Id 这个时候以及可以用id来表示单词 接下来就是embedded 将id映射为词向量 而在transformer里面使用的是位置编码 因此还需要将词向量进一步转化为位置编码(因为tr看的是整个句子 不知道顺序)
```
E_pos = E + PositionalEncoding
```


<img width="494" height="168" alt="image" src="https://github.com/user-attachments/assets/5c167a6f-5873-4b34-b9a6-42addaf61364" />
这是其中位置编码的计算公式

# encoder
完成位置编码后 最为输入进入encoder层 里面带有一个多头注意力 然后进入残差连接和norm 并且有数个相同的子层

### 残差连接 就是把是指把子层（例如注意力层或前馈层）的输入 x 直接加到 子层的输出上：
```
output=𝑥+Sublaye(𝑥)
```

<img width="1013" height="252" alt="image" src="https://github.com/user-attachments/assets/6889c4e8-0bbd-4f61-a650-9ab6e26d3e72" />
### norm  
之前cs231n的时候就学习过 主要是一个均值和方差的求解  使数据更稳定

然后会继续进入一个前馈层 其实就是simple的神经网络(全连接层) 输入-隐藏-输出层 对数据进行处理 然后再次残差连接-norm 经过n次后 传入decoder
# decoder
这个就是生成的过程了 经过了前面对整个句子的学习后 decoder要做的就是从头开始一个个生成单词 但是由于训练的过程中 整个句子都会被看到 因此为了防止过拟合（偷看答案） 因此我们加入了mask机制 遮住整个句子 防止偷看答案 这也跟推理的过程是一致的 防止过拟合 然后decoder里面的结构就跟encoder一样了 经过数次重复后 对输出的结果进行softmax处理 使用argmax得到最好的答案 这样就是整个transformer的结构了

# 注意力机制
实际上之前231n也已经学习过了 这里在讲一遍
三个重要变量（向量）**key**，**value**，**query**，output其实是value的一个加权和

对于每个value  他的权重是又key和query的相似度的来的

最重要的公式也就是

<img width="812" height="180" alt="image" src="https://github.com/user-attachments/assets/a4604db5-ed3e-4388-8818-0ba8a7d1a050" />

其中的 Q V K都是向量 这边使用两次矩阵乘法 使得可以同时计算多个特征（并行性强）

除以d_k(向量距离长) 如果距离太长 最后softmax得到的结果会太靠近两端 这样就会导致梯度的计算过小 模型就跑不动 因此我们选择除以d_k这样使得得到的结果会跟居中

# 使用多头注意力
由于我们进行相似度的求解 如果只使用一个head 就是导致信息被平均化 因此我们使用多头 将向量分成多个子空间 然后分别求相似度 最后再进行合并 就能够使得其具有更好的效果 他们可以具有不同的W权重

# 对比其他模型
<img width="1406" height="282" alt="image" src="https://github.com/user-attachments/assets/a594970f-87bf-4d90-9297-8b5335886d80" />
对于更张的序列来说 使用限制版本的子注意力（只关注部分）能够有所缓解二次的复杂度的缺陷

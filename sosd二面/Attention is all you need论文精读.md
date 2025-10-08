transformer使用编码器解码器架构
编码器 首先将输入的词转化为词向量（长度为n） 然后进行处理 编码器可以看到整个句子
解码器 无法看到整个句子 将解码器的输出做为输入 进行处理 （一个词一个词生成） 同时 在生成yt的时候 y1~yt-1都会作为输入 从而共同生成yt 也叫自回归

<img width="1065" height="890" alt="image" src="https://github.com/user-attachments/assets/8fe29afd-64c8-415f-860c-dab30408795e" />
然后是典中典的 transformer的图 讲滴非常清楚

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        #TODO:按照中所述构造位置编码数组#
        #变压器_标题.ipynb。目标是让每一行交替#
        #正弦和余弦，其指数为0、0、2、2、4、4等，最高可达#
        #embed_dim。当然，这个确切的规格有点武断，但是#
        #这正是签名者所期待的。作为参考，我们的解决方案是#
        #少于5行代码#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        #TODO:索引到位置编码数组中，并添加#
        #适当的输入序列。别忘了申请退学#
        #之后。这应该只需要几行代码#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x+=self.pe[:,:x.size(1),:]
        output=self.dropout(x)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
      
一个模型层，它实现了屏蔽注意力的简化版本，如
由“注意力是你所需要的一切”介绍(https://arxiv.org/abs/1706.03762).

用途：
attn=多头注意（embed_dim，num_heads=2）

#自我关注
data=torch.randn（batch_size，sequence_length，embed_dim）
self_attn_output=attn（查询=数据，键=数据，值=数据）

#使用两个输入进行注意力
other_data=torch.randn（batch_size，sequence_length，embed_dim）
attn_output=attn（查询=数据，键=其他数据，值=其他数据）

    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
          
计算所提供数据的屏蔽注意力输出，计算
所有的注意力都是平行的。

在下面的形状定义中，N是批量大小，S是源
序列长度，T是目标序列长度，E是嵌入
尺寸。

输入：
-query：输入用作查询的数据，形状为（N，S，E）
-key：用作键的输入数据，形状（N、T、E）
-value：用作形状（N、T、E）的值的输入数据
-attn_mask：形状（S，T）的数组，其中mask[i]==0表示标记
源中的i不应影响目标中的令牌j。

退货：
-输出：形状为（N，S，E）的张量，给出以下的加权组合
根据使用键计算的注意力权重，数据值
和查询。
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        #TODO：使用中给出的方程式实现多头注意力#
        #变压器_标题.ipynb#
        #一些提示：#
        #1）你想把你的形状从（N，T，E）分割成（N，T，H，E/H）#
        #其中H是头的数量#
        #2）函数torch.matmul允许您进行批量矩阵乘法#
        #例如，你可以将（N，H，T，E/H）乘以（N，H，E/H，T）得到一个#
        #形状（N、H、T、T）。有关更多示例，请参见#
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #3）对于应用attn_mask，考虑如何将分数修改为#
        #防止值影响输出。具体来说，PyTorch#
        #函数masked_fill可能会派上用场#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        query=query.reshape(N,S,self.n_head,self.head_dim).transpose(1,2)
        key=key.reshape(N,T,self.n_head,self.head_dim).transpose(1,2)
        value=value.reshape(N,T,self.n_head,self.head_dim).transpose(1,2)
        energy=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(self.head_dim)
        if attn_mask is not None:
          energy=energy.masked_fill(attn_mask==0,float('-inf'))
        attention=torch.softmax(energy,dim=3)
        attention=self.attn_drop(attention)
        output=torch.matmul(attention,value).transpose(1,2).reshape(N,S,E)
        output=self.proj(output)
        # Q = self.query(query)
        # K = self.key(key)
        # V = self.value(value)

        # # 获取投影个数
        # H = self.n_head
        # # 获取投影维度
        # D = self.head_dim
        # # 矩阵分割 与 转置 这里需要结合QKV的形状来理解
        # Q = Q.reshape(N, S, H, D).transpose(1, 2)  # (N H S D)
        # K = K.reshape(N, T, H, D).transpose(1, 2)  # (N H T D)
        # V = V.reshape(N, T, H, D).transpose(1, 2)  # (N H T D)

        # # 矩阵乘法算权重  (N, H, S, K) * (N, H, K, T) -> N, H, S, T
        # energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # (N H S T)

        # # 判断是否需要mask
        # if attn_mask is not None:
        #     energy = energy.masked_fill(attn_mask == 0, float('-inf'))

        # # softmax计算
        # A = torch.softmax(energy, dim=3)  # 对第四维度进行softmax

        # # 使用dropout
        # A = self.attn_drop(A)

        # # 计算加权和  (N, H, S, T) * (N, H, T, K) -> (N, H, S, K)
        # Y = A.matmul(V)

        # # 再投影回去
        # Y = Y.transpose(1, 2).reshape(N, S, E)  # (N, S, E)
        # output = self.proj(Y)  # (N, S, E)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output



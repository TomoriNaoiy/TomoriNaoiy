from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        初始化新网络。

        输入：
        -input_dim：给出输入数据大小的元组（C，H，W）
        -num_filters：卷积层中要使用的过滤器数量
        -filter_size：卷积层中使用的过滤器的宽度/高度
        -hidden_dim：在完全连接的隐藏层中使用的单元数
        -num_classes：从最终仿射层生成的分数数量。
        -weight_scale：给出随机初始化标准偏差的标量
        重量。
        -reg：给出L2正则化强度的标量
        -dtype：用于计算的numpy数据类型。
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        #TODO：初始化三层卷积的权重和偏差#
        #网络。权重应从以0.0为中心的高斯函数初始化#
        #标准偏差等于weight_scale；偏见应该是#
        #初始化为零。所有权重和偏差应存储在#
        #字典本身。存储卷积的权重和偏差#
        #使用键“W1”和“b1”分层；使用键“W2”和“b2”#
        #隐藏仿射层的权重和偏差以及关键字“W3”和“b3”#
        #用于输出仿射层的权重和偏差#
        #                                                                          #
        #重要提示：对于此任务，您可以假设填充#
        #选择第一卷积层的步长，以便#
        #**输入的宽度和高度保持不变**。看看#
        #loss（）函数的开头，看看这是如何发生的#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #这里使用2*2max_pool
        c,h,w=input_dim
        H_out=h//2
        W_out=w//2
        self.params["W1"]=weight_scale*np.random.randn(num_filters,c,filter_size,filter_size)
        self.params["b1"]=np.zeros(num_filters)
        self.params["W2"]=weight_scale*np.random.randn(num_filters*H_out*W_out,hidden_dim) 
        self.params["b2"]=np.zeros(hidden_dim)
        self.params["W3"]=weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params["b3"]=np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        #TODO：实现三层卷积网络的前向传递#
            #计算X的课堂分数并将其存储在分数中#
            #变量#
            #                                                                          #
            #请记住，您可以使用cs231n/fast_layers.py中定义的函数和#
            #cs231n/layer_utils.py在您的实现中（已导入）#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param) 
        out2, cache2 = affine_relu_forward(out1, W2, b2)
        scores, cache3 = affine_forward(out2, W3, b3)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        #TODO：实现三层卷积网络的反向传递#
        #将损失和梯度存储在损失和梯度变量中。计算#
        #使用softmax进行数据丢失，并确保grads[k]保持梯度#
        #对于self.params[k]。别忘了添加L2正则化#
        #                                                                          #
        #注意：为确保您的实施与我们的相匹配，并且您通过了#
        #自动化测试，确保你的L2正则化包括一个因素#
        #0.5，以简化梯度的表达式#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss,dscores=softmax_loss(scores,y)
        loss += 0.5* self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2) + 0.5 * self.reg * np.sum(W3 * W3)
        dscore2,dW3,db3=affine_backward(dscores,cache3)
        dscore1,dW2,db2=affine_relu_backward(dscore2,cache2)
        dx,dW1,db1=conv_relu_pool_backward(dscore1,cache1)
        grads["W1"]=dW1+self.reg*W1
        grads["W2"]=dW2+self.reg*W2
        grads["W3"]=dW3+self.reg*W3
        grads["b1"]=db1
        grads["b2"]=db2
        grads["b3"]=db3
       
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

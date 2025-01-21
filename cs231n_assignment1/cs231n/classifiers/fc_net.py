from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    
具有ReLU非线性和非线性的两层全连接神经网络
采用模块化层设计的softmax损耗。我们假设一个输入维度
D，H的隐藏维度，并对C类进行分类。

架构应该是仿射-真实-仿射-softmax。

请注意，此类不实现梯度下降；相反，它
将与负责运行的单独求解器对象交互
优化。

模型的可学习参数存储在字典中
self.params将参数名称映射到numpy数组。

    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        初始化新网络。

输入：
-input_dim：一个整数，表示输入的大小
-hidden_dim：一个整数，表示隐藏层的大小
-num_classes：一个整数，表示要分类的类的数量
-weight_scale：给出随机标准偏差的标量
权重的初始化。
-reg：给出L2正则化强度的标量。
"""
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        #TODO：初始化两层网络的权重和偏差。重量#
#应从以0.0为中心的高斯函数初始化#
#标准偏差等于weight_scale，偏差应为#
#初始化为零。所有权重和偏差应存储在#
#字典self.params，带第一层权重#
#并使用键“W1”和“b1”以及第二层进行偏置#
#使用“W2”和“b2”键进行权重和偏差设置#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params["W1"]=weight_scale*np.random.randn(input_dim,hidden_dim)
        self.params["b1"]=np.zeros(hidden_dim)
        self.params["W2"]=weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params["b2"]=np.zeros(num_classes)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        """
计算小批量数据的损失和梯度。

输入：
-X：形状（N，d_1，…，d_k）的输入数据数组
-y：形状为（N，）的标签数组。y[i]给出了X[i]的标签。

退货：
如果y为None，则运行模型的测试时间向前传递并返回：
-scores：给出分类分数的形状（N，C）数组，其中
scores[i]，c]是X[i]和类c的分类分数。

如果y不为None，则进行训练时间前后传球
返回一个元组：
-loss：给出损失的标量值
-grads:与self.params具有相同键的字典，映射参数
关于这些参数的损失梯度的名称。
"""
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        #TODO：实现两层网络的前向传递，计算#
#X的班级分数，并将其存储在分数变量中#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        X=X.reshape(X.shape[0],-1)
        f=lambda scores:np.maximum(0,scores)
        hidden_scores=f(X.dot(self.params["W1"])+self.params["b1"])
        scores=hidden_scores.dot(self.params["W2"])+self.params["b2"]



        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        #TODO：实现两层网络的向后传递。储存损失#
#在梯度字典中的损失变量和梯度中。计算数据#
#使用softmax进行损失，并确保grads[k]保持梯度#
#self.params[k]。别忘了添加L2正则化#
#                                                                          #
#注意：为确保您的实施与我们的相匹配，并且您通过了#
#自动化测试，确保你的L2正则化包括一个因素#
#0.5，以简化梯度的表达式#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        num_train=X.shape[0]
        scores=(scores.T-np.max(scores,axis=1)).T
        correct_class_score=scores[np.arange(num_train),y]
        loss=np.sum(-correct_class_score+np.log(np.sum(np.exp(scores),axis=1)))
        p=np.exp(scores)/np.sum(np.exp(scores),axis=1).reshape(-1,1)
        p[np.arange(num_train),y]-=1
        loss/=num_train
        loss+=0.5*self.reg*np.sum(self.params["W1"]*self.params["W1"])+0.5*self.reg*np.sum(self.params["W2"]*self.params["W2"])
        #正向传播获得loss
        dscores=p.copy()
        dscores/=num_train
        grads["W2"]=(hidden_scores.T).dot(dscores)+self.reg*self.params["W2"]
        grads["b2"]=np.sum(dscores,axis=0)
        dhidden_scores=dscores.dot(self.params["W2"].T)
        dhidden_scores[hidden_scores<=0]=0
        grads["W1"]=(X.T).dot(dhidden_scores)+self.reg*self.params["W1"]
        grads["b1"]=np.sum(dhidden_scores,axis=0)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

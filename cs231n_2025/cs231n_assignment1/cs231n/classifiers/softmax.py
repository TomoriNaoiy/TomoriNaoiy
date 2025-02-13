from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    """
Softmax损失函数，简单实现（带循环）

输入有维度D，有C类，我们对小批量进行操作
N个例子。

输入：
-W：包含权重的形状（D，C）的numpy数组。
-X：一个包含小批量数据的形状为（N，D）的numpy数组。
-y：一个包含训练标签的形状（N，）的numpy数组；y[i]=c表示
X[i]具有标签c，其中0<=c<c。
-reg：（float）正则化强度

返回一个元组：
-单浮点数损失
-相对于权重W的梯度；与W形状相同的阵列
"""
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes=W.shape[1]
    num_train=X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #TODO：使用显式循环计算softmax损失及其梯度#
#将损失存储在损失中，将梯度存储在dW中。如果你不小心#
#在这里，很容易遇到数值不稳定。别忘了#
#正规化！
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)***** 
    for i in range(num_train):
        score=X[i].dot(W)
        score-=np.max(score)
        correct_class_score=score[y[i]]
        loss+=-correct_class_score+np.log(np.sum(np.exp(score)))
        for j in range(num_classes):
            p=np.exp(score[j])/np.sum(np.exp(score))
            if j==y[i]:
                dW[:,y[i]]+=(p-1)*X[i]
            else:
                dW[:,j]+=p*X[i]
    loss/=num_train
    dW/=num_train
    loss+=reg*np.sum(W*W)
    dW+=reg*W


    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #TODO：不使用显式循环计算softmax损失及其梯度#
#将损失存储在损失中，将梯度存储在dW中。如果你不小心#
#在这里，很容易遇到数值不稳定。别忘了#
#正规化#
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes=W.shape[1]
    num_train=X.shape[0]
    score=X.dot(W)
    score=(score.T-np.max(score,axis=1)).T
    correct_class_score=score[np.arange(num_train),y]
    loss=np.sum(-correct_class_score+np.log(np.sum(np.exp(score),axis=1)))
    p=np.exp(score)/np.sum(np.exp(score),axis=1).reshape(-1,1)
    p[np.arange(num_train),y]-=1
    dW=(X.T).dot(p)
    loss/=num_train
    dW/=num_train
    loss+=reg*np.sum(W*W)
    dW+=reg*W
   

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

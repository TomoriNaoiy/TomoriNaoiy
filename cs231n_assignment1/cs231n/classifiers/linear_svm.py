from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    结构化SVM损失函数，朴素实现（带循环）。

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j]+=X[i]*1
                dW[:,y[i]]+=X[i]*-1

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
   # 现在损失是所有训练示例的总和，但我们想要它
#而是一个平均值，所以我们除以numtrain。
    loss /= num_train
    dW/=num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #待办事项：#
    #计算损失函数的梯度并将其存储为dW#
    #与其先计算损失，然后计算导数#
    #在计算导数的同时#
    #损失正在计算中。因此，您可能需要修改一些#
    #上面的代码用于计算梯度#
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW+=2*reg*W
    #其余见上面代码

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #待办事项：#
    #实现结构化SVM损失的矢量化版本，存储#
    #导致损失#
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    score=X.dot(W)
    correct_class_score=score[np.arange(num_train),y].reshape(-1,1)
    margin=np.maximum(0,score-correct_class_score+1)
    margin[np.arange(num_train),y]=0
    loss=np.sum(margin)/num_train
    loss+=reg*np.sum(W * W)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #待办事项：#
    #为结构化SVM实现梯度的矢量化版本#
    #损失，将结果存储在dW中#
    #                                                                           #
    #提示：与其从头开始计算梯度，它可能更容易#
    #重用一些用于计算的中间值#
    #损失#
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW = np.zeros(W.shape)
    gradient=margin.copy()
    gradient[margin>0]=1
    row_num=np.count_nonzero(margin > 0, axis=1)
    gradient[np.arange(num_train),y]=-row_num
    dW=(X.T).dot(gradient)/num_train
    dW+=2*reg*W

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

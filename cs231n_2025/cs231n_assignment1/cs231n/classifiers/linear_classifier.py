from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from ..classifiers.linear_svm import *
from ..classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        '''        使用随机梯度下降训练这个线性分类器。

输入：
-X：一个包含训练数据的形状为（N，D）的numpy数组；有N
每个维度D的训练样本。
-y：一个包含训练标签的形状（N，）的numpy数组；y[i]=c
对于c类，X[i]的标签为0<=c<c。
-learning_rate：用于优化的（浮动）学习率。
-reg：（float）正则化强度。
-num_iters：（整数）优化时要采取的步骤数
-batch_size：（整数）每一步要使用的训练示例数。
-verbose：（boolean）如果为true，则在优化过程中打印进度。

输出：
包含每次训练迭代时损失函数值的列表。
'''
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        #运行随机梯度下降以优化W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #待办事项：#
            #训练数据中的样本batch_size元素及其#
            #在这一轮梯度下降中使用相应的标签#
            #将数据存储在X_batch中，并将其对应的标签存储在#
            #y_batch；采样后，X_batch应具有形状（batch_size，dim）#
            #y_batch应该具有形状（batch_size，）#
            #                                                                       #
            #提示：使用np.random.choice生成索引。取样#
            #替换比不替换的采样更快#
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            batch_index=np.random.choice(num_train,batch_size,replace=True)
            X_batch=X[batch_index,:]
            y_batch=y[batch_index]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #待办事项：#
            #使用梯度和学习率更新权重。
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.W -= learning_rate * grad

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        
        '
        使用此线性分类器的训练权重来预测标签
数据点。

输入：
-X：一个包含训练数据的形状为（N，D）的numpy数组；有N
每个维度D的训练样本。

退货：
-y_pred：X中数据的预测标签y_pred是一维的
长度为N的数组，每个元素都是一个整数，给出预测值
类。
"""

        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        score=X.dot(self.W)
        y_pred=np.argmax(score,axis=1)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        """
计算损失函数及其导数。
子类将覆盖此内容。

输入：
-X_batch：一个形状为（N，D）的numpy数组，包含一个N的小批量
数据点；每个点具有维度D。
-y_batch：一个包含小批量标签的形状（N，）的numpy数组。
-reg：（float）正则化强度。

返回：一个包含以下内容的元组：
-损失作为单一浮动
-相对于自我的梯度。W、 与W形状相同的阵列
"""
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
            训练分类器。对于k近邻，这只是
            记忆训练数据。

            输入：
            -X：一个包含训练数据的numpy形状数组（num_train，D）
            由每个维度为D的num_train样本组成。
            -y：一个包含训练标签的形状（N，）的numpy数组，其中
            y[i]是X[i]的标签。
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
          使用此分类器预测测试数据的标签。

            输入：
            -X：一个包含测试数据的numpy形状数组（num_test，D），由以下部分组成
            每个维度D都有num_test样本。
            -k：投票支持预测标签的最近邻数量。
            -num_loops：确定使用哪个实现来计算距离
            在训练点和测试点之间。

            退货：
            -y：一个包含预测标签的numpy形状数组（num_test，）
            其中y[i]是测试点X[i]的预测标签。
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
          计算X中每个测试点与每个训练点之间的距离
        在自我。X_train使用嵌套循环对训练数据和
        测试数据。

        输入：
        -X：一个包含测试数据的numpy形状数组（num_test，D）。

        退货：
        -dists：一个numpy形状数组（num_test，num_train），其中dists[i]
        是第i个测试点和第j个训练点之间的欧几里德距离
        点.第一点X[i]。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm()
                # 计算第i个测试点和第j个测试点之间的l2距离#
                #训练点，并将结果存储在dists[i]，j]中。你应该#
                #不要在维度上使用循环，也不要使用np.linalg.norm（）。.          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                dists[i][j]=np.sqrt(np.sum(np.square(self.X_train[j]-X[i])))


                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        计算X中每个测试点与每个训练点之间的距离
        在自我。X_train使用单个循环对测试数据进行处理。

        输入/输出：与compute_distance_two_loops相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #计算第i个测试点和所有训练点之间的l2距离#
            #点，并将结果存储在dists[i]，：]中#
            #不要使用np.linalg.norm（）#
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            dists[i,:]=np.sqrt(np.sum(np.square(self.X_train-X[i]),axis=1))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #计算所有测试点和所有训练之间的l2距离#
            #点，而不使用任何显式循环，并将结果存储在#
            #dists#
            #                                                                       #
            #您应该只使用基本的数组操作来实现此函数#
            #特别是你不应该使用scipy的函数#
            #也不要使用np.linalg.norm（）#
            #                                                                       #
            #提示：尝试使用矩阵乘法来表述l2距离#
            #以及两个广播总和。
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        X_2=np.sum(X**2,axis=1,keepdims=True)
        X_train_2=np.sum(self.X_train**2,axis=1)
        X_train_2 = X_train_2.reshape(1, -1)
        dists=np.sqrt(X_2-2*np.dot(X,self.X_train.T)+X_train_2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
          给定测试点和训练点之间的距离矩阵，
        预测每个测试点的标签。

        输入：
        -dists：一个numpy形状数组（num_test，num_train），其中dists[i]
        给出了第i个测试点和第j个训练点之间的距离。

        退货：
        -y：一个包含预测标签的numpy形状数组（num_test，）
        其中y[i]是测试点X[i]的预测标签。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #使用距离矩阵找到第i个最近邻居的k个#
            #测试点，并使用self.y_train查找这些标签#
            #邻居。将这些标签放在最靠近的地方#
            #提示：查找函数numpy.argsort。
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
           
            closest_y=self.y_train[np.argsort(dists[i])[:k]]
            

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #现在你已经找到了k个最近邻居的标签，你#
            #需要在标签列表中找到最常见的标签#
            #将此标签存储在y_pred[i]中。通过选择较小的来打破关系#
            #标签。#
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            closest_y_unique, label_number = np.unique(closest_y, return_counts=True)
            y_pred[i]=closest_y_unique[np.argmax(label_number)]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred

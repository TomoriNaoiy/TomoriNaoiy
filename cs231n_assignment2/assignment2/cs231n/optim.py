import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
此文件实现了各种常用的一阶更新规则
用于训练神经网络。每个更新规则都接受当前权重和
损失相对于这些权重的梯度，并产生下一组
重量。每个更新规则都有相同的接口：

def更新（w，dw，config=无）：

输入：
-w：一个给出当前权重的numpy数组。
-dw：一个与w形状相同的numpy数组，给出
关于w的损失。
-config：包含学习等超参数值的字典
速率、动量等。如果更新规则要求缓存多个值
迭代后，config也将保存这些缓存值。

退货：
-next_w：更新后的下一个点。
-config：要传递给下一次迭代的配置字典
更新规则。

注意：对于大多数更新规则，默认学习率可能不会
表现良好；然而，其他超参数的默认值应该
适用于各种不同的问题。

为了提高效率，更新规则可以执行就地更新，修改w和
将next_w设置为等于w。

"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
      利用动量执行随机梯度下降。

    配置格式：
    -learning_rate：标量学习率。
    -动量：0到1之间的标量，给出动量值。
    设置动量=0会减少到sgd。
    -velocity：一个与w和dw形状相同的numpy数组，用于存储
    梯度的移动平均值。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    #TODO：实现动量更新公式。将更新后的值存储在#
    #nextw变量。您还应该使用和更新速度v#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    v=config["momentum"]*v-config["learning_rate"]*dw
    next_w=w+v

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    使用RMSProp更新规则，该规则使用平方的移动平均值
    梯度值，以设置自适应的每参数学习率。

    配置格式：
    -learning_rate：标量学习率。
    -decay_rate：0到1之间的标量，给出平方的衰减率
    梯度缓存。
    -epsilon：用于平滑的小标量，以避免除以零。
    -cache：梯度二阶矩的移动平均值。
    """

    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    #TODO:实现RMSprop更新公式，存储w的下一个值#
  #在nextw变量中。不要忘记更新存储在中的缓存值#
  #配置[缓存]#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cache=config["cache"]
    cache=config["decay_rate"]*cache+(1-config["decay_rate"])*dw**2
    next_w=w-config["learning_rate"]*dw/(np.sqrt(cache)+config["epsilon"])
    config["cache"]=cache
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    
    
使用Adam更新规则，该规则结合了
梯度及其平方和偏差校正项。

配置格式：
-learning_rate：标量学习率。
-beta1：梯度一阶矩移动平均值的衰减率。
-β2：梯度二阶矩移动平均值的衰减率。
-epsilon：用于平滑的小标量，以避免除以零。
-m：梯度移动平均值。
-v：梯度平方的移动平均值。
-t：迭代次数。
"""
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #ss
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    #TODO:实现Adam更新公式，将w的下一个值存储在#
#nextw变量。别忘了更新m、v和t变量#
#存储在config中#
#                                                                         #
#注意：为了匹配参考输出，请修改t _before_#
#在任何计算中使用它#

    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)***** 
    m=config["m"]
    v=config["v"]
    t=config["t"]
    t+=1  
    m=config["beta1"]*m+(1-config["beta1"])*dw
    mt=m/(1-config["beta1"]**t)
    v=config["beta2"]*v+(1-config["beta2"])*dw**2
    vt=v/(1-config["beta2"]**t)
    next_w=w-config["learning_rate"]*mt/(np.sqrt(vt)+config["epsilon"])
    config["m"]=m
    config["v"]=v
    config["t"]=t
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config

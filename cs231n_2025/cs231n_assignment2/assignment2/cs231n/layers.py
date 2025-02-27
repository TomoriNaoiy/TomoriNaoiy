from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x.reshape((x.shape[0],-1)).dot(w) + b


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dx = dout.dot(w.T).reshape(x.shape)
    dw = (x.reshape(x.shape[0],-1).T).dot(dout)
    db = np.sum(dout, axis=0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out=np.maximum(0,x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx=dout*(x>0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train=x.shape[0]
    x=(x.T-np.max(x,axis=1)).T
    correct_class_score=x[np.arange(num_train),y]
    loss=np.sum(-correct_class_score+np.log(np.sum(np.exp(x),axis=1)))
    p=np.exp(x)/np.sum(np.exp(x),axis=1).reshape(-1,1)
    p[np.arange(num_train),y]-=1
    dx=p
    loss/=num_train
    dx/=num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    '''Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    
    批标准化的转发。

在训练过程中，样本均值和（未校正的）样本方差为
根据小批量统计数据计算，用于规范化传入数据。
在训练过程中，我们还保持了指数级衰减的跑步平均值
每个特征的均值和方差，这些均值用于归一化
测试时的数据。

在每个时间步，我们使用以下公式更新均值和方差的运行平均值
基于动量参数的指数衰减：

running_man=动量*running_man+（1-动量）*sample_man
running_var=动量*running_var+（1-动量）*sample_var

请注意，批归一化纸张建议使用不同的测试时间
行为：他们使用
使用大量训练图像而不是使用运行平均值。For
在这个实现中，我们选择使用运行平均值，因为
它们不需要额外的估计步骤；火炬7
批处理规范化的实现也使用运行平均值。

输入：
-x：形状数据（N，D）
-gamma：形状的比例参数（D，）
-β：形状偏移参数（D，）
-bn_param：具有以下键的字典：
-模式：“训练”或“测试”；必修的
-eps：数值稳定性常数
-动量：运行均值/方差的常数。
-running_man：形状（D，）的数组，给出特征的运行均值
-running_var形状（D，）的数组，给出特征的运行方差

返回一个元组：
-out：变形（N，D）
-cache：反向传递中所需的值元组
    '''
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #TODO：实施批量规范的培训时间向前传递#
        #使用小批量统计来计算均值和方差，使用#
        #这些统计数据用于规范传入数据，并进行缩放和#
        #使用gamma和beta对归一化数据进行移位#
        #                                                                     #
        #您应该将输出存储在变量out中。任何中间体#
        #反向传递所需的信息应存储在缓存中#
        #变量#
        #                                                                     #
        #您还应该同时使用计算出的样本均值和方差#
        #使用动量变量更新跑步平均值和跑步#
        #variance，将结果存储在running_man和running_var中#
        #变量#
        #                                                                     #
        #请注意，虽然你应该跟踪跑步情况#
        #方差，您应该根据标准对数据进行归一化#
        #偏差（方差平方根）代替#
        #参考原始论文(https://arxiv.org/abs/1502.03167)   #
        #可能被证明是有帮助的#
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)****
        x_mean = np.mean(x, axis=0)
        x_var=np.var(x,axis=0)
        x_std=np.sqrt(x_var+eps)
        x_norm=(x-x_mean)/x_std
        out=gamma*x_norm+beta
        cache=(x,x_mean,x_var,x_std,x_norm,out,gamma,beta,eps)
        running_mean=momentum*x_mean+(1-momentum)*running_mean
        running_var=momentum*x_var+(1-momentum)*running_var
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #TODO：实现批标准化的测试时间传递#
        #使用运行均值和方差对传入数据进行归一化#
        #然后使用gamma和beta对归一化数据进行缩放和移位#
        #将结果存储在out变量中#
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out=gamma*((x-running_mean)/np.sqrt(running_var+eps))+beta


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    批量标准化的反向传递。

对于此实现，您应该为以下内容编写一个计算图
在纸张上进行批量归一化，并通过反向传播梯度
中间节点。

输入：
-dout：上游导数，形状（N，D）
-cache：batchnorm_forward中的中间体变量。

返回一个元组：
-dx：相对于输入x的梯度，形状为（N，D）
-dgamma：相对于比例参数gamma的梯度，形状（D，）
-dbeta：相对于形状（D，）的偏移参数β的梯度

    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    #TODO：实现批处理规范化的反向传递。存储#
    #结果为dx、dgamma和dbeta变量#
    #参考原始论文(https://arxiv.org/abs/1502.03167)       #
    #可能被证明是有帮助的#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x,x_mean,x_var,x_std,x_norm,out,gamma,beta,eps=cache
    dgamma=np.sum(dout*x_norm,axis=0)
    dbeta=np.sum(dout,axis=0)
    dx_norm=dout*gamma
    dx_var=np.sum(dx_norm*(x-x_mean)*(-0.5)*np.power(x_var+eps,-1.5),axis=0)
    dx_mean=np.sum(dx_norm*(-1)/x_std,axis=0)+dx_var*np.sum(-2*(x-x_mean),axis=0)/x.shape[0]
    dx=dx_norm/x_std+dx_var*2*(x-x_mean)/x.shape[0]+dx_mean/x.shape[0]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 

    # Unpack cache
    x,x_mean,x_var,x_std,x_norm,out,gamma,beta,eps=cache
    dgamma=np.sum(dout*x_norm,axis=0)
    dbeta=np.sum(dout,axis=0)
    dx_norm=dout*gamma
    dx_var=np.sum(dx_norm*(x-x_mean)*(-0.5)*np.power(x_var+eps,-1.5),axis=0)
    dx_mean=np.sum(dx_norm*(-1)/x_std,axis=0)+dx_var*np.sum(-2*(x-x_mean),axis=0)/x.shape[0]
    dx=dx_norm/x_std+dx_var*2*(x-x_mean)/x.shape[0]+dx_mean/x.shape[0]
 
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    #TODO：实施层规范的培训时间向前传递#
    #对传入数据进行标准化，并对标准化数据进行缩放和移动#
    #使用gamma和beta#
    #提示：这可以通过稍微修改你的训练时间来实现#
    #实现批处理规范化，并插入一两行#
    #放置得当的代码。特别是，你能想到任何矩阵吗#
    #您可以执行的转换，这将使您能够复制#
    #批量规范代码几乎保持不变#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x_mean = np.mean(x, axis=1).reshape(-1,1)
    x_var=np.var(x,axis=1).reshape(-1,1)
    x_std=np.sqrt(x_var+eps)
    x_norm=(x-x_mean)/x_std
    out=gamma*x_norm+beta
    cache=(x,x_mean,x_var,x_std,x_norm,out,gamma,beta,eps)
    




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    x,x_mean,x_var,x_std,x_norm,out,gamma,beta,eps=cache
    dgamma=np.sum(dout*x_norm,axis=0)
    dbeta=np.sum(dout,axis=0)
    dx_norm=dout*gamma
    dx_var=np.sum(dx_norm*(x-x_mean)*(-0.5)*np.power(x_var+eps,-1.5),axis=1).reshape(-1,1)
    dx_mean=np.sum(dx_norm*(-1)/x_std,axis=1).reshape(-1,1)+dx_var*np.sum(-2*(x-x_mean),axis=1).reshape(-1,1)/x.shape[1]
    dx=dx_norm/x_std+dx_var*2*(x-x_mean)/x.shape[1]+dx_mean/x.shape[1]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mask=(np.random.randn(*x.shape)<p)/p#查阅ai 这里的*x.shape*是解包的用法 例如将（N,D）转化为N，D传给这个方法
        out=x*mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out=x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dx=dout*mask
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    卷积层前向传递的简单实现。

输入由N个数据点组成，每个数据点有C个通道，高度H和
宽度W。我们用F个不同的滤波器对每个输入进行卷积，其中每个滤波器
跨越所有C通道，高度为HH，宽度为WW。

输入：
-x：形状（N、C、H、W）的输入数据
-w：形状的过滤器权重（F、C、HH、WW）
-b：形状偏差（F，）
-conv_param：具有以下键的字典：
-“步幅”：大脑中相邻感受野之间的像素数
水平和垂直方向。
-“pad”：用于对输入进行零填充的像素数。

在填充过程中，“填充”零应对称放置（即两侧相等）
沿着输入的高度和宽度轴。小心不要修改原作
直接输入x。

返回一个元组：
-out：输出数据，形状为（N，F，H'，W'），其中H'和W'由下式给出
H'=1+（H+2*垫-HH）/步幅
W'=1+（W+2*垫-WW）/步幅
-缓存：（x，w，b，conv_param）

    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W=x.shape
    F,C,HH,WW=w.shape
    stride=conv_param["stride"]
    pad=conv_param["pad"]
    H_out=1+(H+2*pad-HH)//stride
    W_out=1+(W+2*pad-WW)//stride
    out=np.zeros((N,F,H_out,W_out))
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant", constant_values=0)
    w_row=w.reshape(F,-1)
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    window=x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW].reshape(1,-1)
                    out[n, f, i, j] = np.sum(window * w_row[f, :]) + b[f]

    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x,w,b,conv_param=cache
    N,C,H,W=x.shape
    F,C,HH,WW=w.shape
    stride=conv_param["stride"]
    pad=conv_param["pad"]
    H_out=1+(H+2*pad-HH)//stride
    W_out=1+(W+2*pad-WW)//stride
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant", constant_values=0)
    dx_pad=np.zeros_like(x_pad)
    dx=np.zeros_like(x)
    dw=np.zeros_like(w)
    db=np.zeros_like(b)
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    window=x_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW]
                    db[f]+=dout[n,f,i,j]
                    dw[f]+=window*dout[n,f,i,j]
                    dx_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW]+=w[f]*dout[n,f,i,j]
                    
    dx = dx_pad[:, :, pad: - pad, pad: - pad]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    最大池化层的前向传递的简单实现。

输入：
-x：输入数据，形状（N、C、H、W）
-pool_param：具有以下键的字典：
-'pool_height'：每个池区域的高度
-'pool_width'：每个池区域的宽度
-“步幅”：相邻池区域之间的距离

这里不需要填充，例如你可以假设：
-（H-池高）步幅百分比==0
-（W-池宽度）步幅百分比==0

返回一个元组：
-out：输出数据，形状为（N，C，H'，W'），其中H'和W'由下式给出
H'=1+（H-台球高度）/步幅
W'=1+（W-池宽）/步幅
-缓存：（x，pool_param）
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W=x.shape
    pool_height=pool_param["pool_height"]
    pool_width=pool_param["pool_width"]
    stride=pool_param["stride"]
    H_out=1+(H-pool_height)//stride
    W_out=1+(W-pool_width)//stride
    out=np.zeros((N,C,H_out,W_out))
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                   window=x[n,c,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width]
                   out[n,c,i,j]=np.max(window)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x,pool_param=cache
    N,C,H,W=x.shape
    pool_height=pool_param["pool_height"]
    pool_width=pool_param["pool_width"]
    stride=pool_param["stride"]
    H_out=1+(H-pool_height)//stride
    W_out=1+(W-pool_width)//stride
    dx=np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                   window=x[n,c,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width]
                   window_index=np.argmax(window)
                   dx[n,c,i*stride+window_index//pool_width,j*stride+window_index%pool_width]+=dout[n,c,i,j]
                   


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W=x.shape
    x=np.moveaxis(x,1,-1).reshape(-1,C)
    out,cache=batchnorm_forward(x,gamma,beta,bn_param)
    out=out.reshape(N,H,W,C)
    out=np.moveaxis(out,-1,1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W=dout.shape
    dout=np.moveaxis(dout,1,-1).reshape(-1,C)
    dx,dgamma,dbeta=batchnorm_backward(dout,cache)
    dx=dx.reshape(N,H,W,C)
    dx=np.moveaxis(dx,-1,1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W=x.shape
    x=x.reshape(N,G,C//G,H,W)
    x_mean=np.mean(x,axis=(2,3,4),keepdims=True)
    x_var=np.var(x,axis=(2,3,4),keepdims=True)
    x_std=np.sqrt(x_var+eps)
    x_norm=(x-x_mean)/x_std
    x_norm=x_norm.reshape(N,C,H,W)
    out=gamma*x_norm+beta
    cache=(x,x_mean,x_var,x_std,x_norm,out,gamma,beta,eps,G)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x,x_mean,x_var,x_std,x_norm,out,gamma,beta,eps,G=cache
    N,C,H,W=dout.shape
    x=x.reshape(N,G,C//G,H,W)
    
    m=C//G*H*W
    dgamma=np.sum(dout*x_norm,axis=(0,2,3),keepdims=True)
    dbeta=np.sum(dout,axis=(0,2,3),keepdims=True)
    dx_norm=(dout*gamma).reshape(N,G,C//G,H,W)
    dx_var=np.sum(dx_norm*(x-x_mean)*(-0.5)*np.power(x_var+eps,-1.5),axis=(2,3,4),keepdims=True)
    dx_mean=np.sum(dx_norm*(-1)/x_std,axis=(2,3,4),keepdims=True)+dx_var*np.sum(-2*(x-x_mean),axis=(2,3,4),keepdims=True)/m
    dx=dx_norm/x_std+dx_var*2*(x-x_mean)/m+dx_mean/m
    dx=dx.reshape(N,C,H,W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta

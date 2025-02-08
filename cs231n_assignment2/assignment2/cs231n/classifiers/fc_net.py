from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *
def affine_bn_relu_forward(x,w,b,gamma,beta,bn_param):
    a,fc_cache=affine_forward(x,w,b)
    bn_out,bn_cache= batchnorm_forward(a,gamma,beta,bn_param)
    out,relu_cache= relu_forward(bn_out)
    cache=(fc_cache,bn_cache,relu_cache)
    return out,cache

def affine_bn_relu_backward(dout,cache):
    fc_cache,bn_cache,relu_cache=cache
    dr=relu_backward(dout,relu_cache)
    dbn,dgamma,dbeta=batchnorm_backward(dr,bn_cache) 
    dx,dw,db=affine_backward(dbn,fc_cache)
    return dx,dw,db,dgamma,dbeta

def affine_ln_relu_forward(x,w,b,gamma,beta,bn_param):
    a,fc_cache=affine_forward(x,w,b)
    ln_out,ln_cache= layernorm_forward(a,gamma,beta,bn_param)
    out,relu_cache= relu_forward(ln_out)
    cache=(fc_cache,ln_cache,relu_cache)
    return out,cache

def affine_ln_relu_backward(dout,cache):
    fc_cache,ln_cache,relu_cache=cache
    dr=relu_backward(dout,relu_cache)
    dln,dgamma,dbeta=layernorm_backward(dr,ln_cache) 
    dx,dw,db=affine_backward(dln,fc_cache)
    return dx,dw,db,dgamma,dbeta

class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    “”类用于多层全连接神经网络。

网络包含任意数量的隐藏层、ReLU非线性，
以及softmax损失函数。这也将实现dropout和批处理/层
规范化作为选项。对于具有L层的网络，架构将是

{仿射-[批/层范数]-relu-[dropout]}x（L-1）-仿射-softmax

其中批处理/层规范化和dropout是可选的，{…}块是
重复L-1次。

可学习的参数存储在self.params字典中，并将被学习
使用求解器类。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
            “”初始化新的FullyConnectedNet。

输入：
-hidden_dims：给出每个隐藏层大小的整数列表。
-input_dim：一个整数，表示输入的大小。
-num_classes：一个整数，表示要分类的类的数量。
-dropout_keep_ratio：介于0和1之间的标量，给出dropout强度。
如果dropout_keep_ratio=1，则网络根本不应该使用dropout。
-规范化：网络应该使用什么类型的规范化。有效值
是“batchnorm”、“layernorm”或“None”表示没有规范化（默认值）。
-reg：给出L2正则化强度的标量。
-weight_scale：给出随机标准偏差的标量
权重的初始化。
-dtype：numpy数据类型对象；所有计算将使用
这种数据类型。float32更快，但精度较低，因此您应该使用
float64用于数字梯度检查。
-seed：如果不是None，则将此随机种子传递给dropout层。
这将使漏失层恶化，因此我们可以对模型进行梯度检查。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        #TODO:初始化网络参数，将所有值存储在#
#self.params词典。存储第一层的权重和偏差#
#在W1和b1中；对于第二层使用W2和b2等。重量应为#
#用标准从以0为中心的正态分布初始化#
#偏差等于weight_scale。偏差应初始化为零#
#                                                                          #
#使用批归一化时，存储#
#gamma1和beta1的第一层；对于第二层，使用gamma2和#
#β2等。缩放参数应初始化为1并移位#
#参数应初始化为零#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        pre_hidden_dim = input_dim
        for i in range(self.num_layers - 1):
            hidden_dim = hidden_dims[i]
            self.params[f'W{i+1}'] = weight_scale * np.random.randn(pre_hidden_dim, hidden_dim)
            self.params[f'b{i+1}'] = np.zeros(hidden_dim)
            pre_hidden_dim = hidden_dim
            # bach Normalization Starts
            if self.normalization == "batchnorm" or self.normalization == "layernorm":
                self.params[f'gamma{i+1}'] = np.ones(hidden_dim)
                self.params[f'beta{i+1}']  = np.zeros(hidden_dim)
            
        self.params[f'W{self.num_layers}'] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
        self.params[f'b{self.num_layers}'] = np.zeros(num_classes)

        # layer_dims = [input_dim] + hidden_dims + [num_classes]
        #   
        # for i in range(self.num_layers):
        #     self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, size=(layer_dims[i], layer_dims[i + 1]))
        #     self.params['b' + str(i + 1)] = np.zeros(layer_dims[i + 1])
   
        #     if self.normalization in ['batchnorm', 'layernorm'] and i != self.num_layers-1: 
        #         self.params[f'gamma{i+1}'] = np.ones(layer_dims[i + 1])
        #         self.params[f'beta{i+1}'] = np.zeros(layer_dims[i + 1])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        #使用dropout时，我们需要将dropout_param字典传递给每个
#dropout层，以便该层知道dropout概率和模式
#（训练/测试）。您可以将相同的dropout_param传递给每个dropout层。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        #通过批量标准化，我们需要跟踪运行方式和
        #方差，因此我们需要向每个批传递一个特殊的bn_param对象
        #标准化层。您应该将self.bn_plans[0]传递给前进传球
        #将第一批标准化层self.bn_plans[1]向前移动
        #第二批归一化层的通过等。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
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
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        #TODO：实现全连接网络的前向传递，计算#
        #X的班级分数并将其存储在分数变量中#
        #                                                                          #
        #使用dropout时，您需要将self.dropout_param传递给每个#
        #放弃前进传球#
        #                                                                          #
        #使用批规范化时，您需要将self.bn_plans[0]传递给#
        #第一批归一化层的前向传递，pass#
        #self.bn_plans[1]到第二批标准化的前向传递#
        #层等#
        ############################################################################
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        layer_input=X
        caches={}
        for i in range(self.num_layers-1):
            W=self.params[f"W{i+1}"]
            b=self.params[f"b{i+1}"]
            if self.normalization in ['batchnorm', 'layernorm']:
                gamma=self.params[f"gamma{i+1}"]
                beta=self.params[f"beta{i+1}"]
                if self.normalization == 'batchnorm':
                    layer_input, caches["bn_cache"+str(i+1)] = affine_bn_relu_forward(layer_input, W, b, gamma, beta, self.bn_params[i])
                if self.normalization == 'layernorm':
                    layer_input, caches["ln_cache"+str(i+1)] = affine_ln_relu_forward(layer_input, W, b, gamma, beta, self.bn_params[i])
            else:
                affine_out, affine_cache = affine_forward(layer_input, W, b)
                relu_out, relu_cache = relu_forward(affine_out)
                caches['affine_cache' + str(i+1)] = affine_cache
                caches['relu_cache' + str(i+1)] = relu_cache
                layer_input=relu_out
            if self.use_dropout:
                layer_input, dropout_cache = dropout_forward(layer_input, self.dropout_param)
                caches['dropout_cache' + str(i+1)] = dropout_cache
        W=self.params[f"W{self.num_layers}"]
        b=self.params[f"b{self.num_layers}"]
        scores,affine_cache=affine_forward(layer_input,W,b)
        caches['affine_cache' + str(self.num_layers)] = affine_cache

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        #TODO：为完全连接的网络实现反向传递。存储#
        #损失变量和梯度中的损失。计算#
        #使用softmax进行数据丢失，并确保grads[k]保持梯度#
        #对于self.params[k]。别忘了添加L2正则化#
        #                                                                          #
        #使用批处理/层规范化时，您不需要对#
        #缩放和移位参数#
        #                                                                          #
        #注意：为确保您的实施与我们的相匹配，并且您通过了#
        #自动化测试，确保你的L2正则化包括一个因素#
        #0.5，以简化梯度的表达式#
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss,dscores=softmax_loss(scores,y)
        W=self.params[f"W{self.num_layers}"]
        affine_cache=caches['affine_cache' + str(self.num_layers)]
        d_relu_out, dW, db = affine_backward(dscores, affine_cache)
        grads[f"W{self.num_layers}"]=dW+self.reg*W
        grads[f"b{self.num_layers}"]=db
        for i in range(self.num_layers-1,0,-1):
            if self.use_dropout:
                d_relu_out=dropout_backward(d_relu_out,caches['dropout_cache' + str(i)])
            if self.normalization == 'batchnorm':
                W=self.params[f"W{i}"]
                bn_cache = caches['bn_cache' + str(i)]
                d_affine_out, dW, db, dgamma, dbeta = affine_bn_relu_backward(d_relu_out, bn_cache)
                grads[f"W{i}"]=dW+self.reg*W
                grads[f"b{i}"]=db
                grads[f"gamma{i}"]=dgamma
                grads[f"beta{i}"]=dbeta
                d_relu_out=d_affine_out
            elif self.normalization == 'layernorm':
                W=self.params[f"W{i}"]
                ln_cache = caches['ln_cache' + str(i)]
                d_affine_out, dW, db, dgamma, dbeta = affine_ln_relu_backward(d_relu_out, ln_cache)
                grads[f"W{i}"]=dW+self.reg*W
                grads[f"b{i}"]=db
                grads[f"gamma{i}"]=dgamma
                grads[f"beta{i}"]=dbeta
                d_relu_out=d_affine_out
            else:
                W=self.params[f"W{i}"]
                affine_cache = caches['affine_cache' + str(i)]
                relu_cache = caches['relu_cache' + str(i)]
                d_affine_out = relu_backward(d_relu_out, relu_cache)
                d_relu_out, dW, db = affine_backward(d_affine_out, affine_cache)
                grads[f"W{i}"]=dW+self.reg*W
                grads[f"b{i}"]=db
           
        for i in range(1, self.num_layers + 1):
            W = self.params['W' + str(i)]
            loss += 0.5 * self.reg * np.sum(W * W)        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

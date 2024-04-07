import numpy as np


def xavier_init(shape):
    """Xavier初始化"""
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * scale


class layer:
    # 一层网络
    def __init__(self, in_feats, out_feats, activation=None, weights=None, bias=None):
        # 权重矩阵
        if weights:
            self.weights = weights
        else:
            self.weights = xavier_init((in_feats, out_feats))
        # 偏置
        if bias:
            self.bias = bias
        else:
            self.bias = np.zeros(out_feats)
        # 激活函数
        self.activation = activation

        # 激活函数输出
        self.activation_output = None

        # 后续层对输出求导的梯度
        self.error = None
        # 输出对激活函数求导后的梯度（最终梯度）
        self.delta = None

    def activate(self, X):
        r = np.dot(X, self.weights) + self.bias
        self.activation_output = self._apply_activation(r)
        return self.activation_output

    def _apply_activation(self, r):
        # 激活函数
        if not self.activation:
            return r
        if self.activation == "relu":
            return np.maximum(r, 0)
        if self.activation == "tanh":
            return np.tanh(r)
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        # 计算激活函数的导数
        if not self.activation:
            return np.ones_like(r)
        elif self.activation == "relu":
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.0
            grad[r <= 0] = 0.0
            return grad
        elif self.activation == "tanh":
            return 1 - r**2
        elif self.activation == "sigmoid":
            return r * (1 - r)
        else:
            raise NotImplementedError

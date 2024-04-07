import numpy as np
from sklearn.metrics import accuracy_score
from layer import layer


class MLP:
    def __init__(self, in_feats, hid_feats, out_feats, num_layer):
        self.layers = []
        for i in range(num_layer):
            if i == 0:
                self.layers.append(layer(in_feats, hid_feats, activation="relu"))
            elif i == num_layer - 1:
                self.layers.append(layer(hid_feats, out_feats))
            else:
                self.layers.append(layer(hid_feats, hid_feats, activation="relu"))

    def forward_propagation(self, X):
        for layer in self.layers:
            X = layer.activate(X)
        return X

    def back_propagation(self, X, y, rate):
        output = self.forward_propagation(X)

        # 分类任务，使用softmax得到类别概率
        prob = np.exp(output - output.max(axis=-1, keepdims=True))
        prob = prob / prob.sum(axis=-1, keepdims=True)

        # 反向计算，保存每层参数的delta，用于更新前一层的参数
        for i in range(len(self.layers))[-1::-1]:
            layer = self.layers[i]
            if layer == self.layers[-1]:
                # 输出层

                # error 为损失函数(softmax交叉熵)对输出求导
                layer.error = prob - y

                # delta 损失函数(softmax交叉熵)对输出求导 * 输出对激活函数的导数
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                # 隐含层
                next_layer = self.layers[i + 1]
                # 根据链式求导法则，隐含层的 error 为反向传播的前一层(正向传播的下一层) delta * 参数w
                layer.error = np.dot(next_layer.weights, next_layer.delta)

                # 再对当前层的激活函数求导
                layer.delta = layer.error * layer.apply_activation_derivative(
                    layer.activation_output
                )

        # 前向传播得到了每层的输出，反向计算得到了每层的 delta
        # 由链式求导法则化简，损失函数对每个参数w的偏导数 = (参数矩阵W的输入 \times delta).T
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                # 第一个隐含层的输入为样本特征X
                output = np.atleast_2d(X)
            else:
                output = np.atleast_2d(self.layers[i - 1].activation_output)

            # 更新参数矩阵
            # 梯度下降算法的优化方向是梯度的反方向，所以：
            # 参数矩阵 = 当前参数矩阵 - 梯度矩阵*学习率
            layer.weights -= np.dot(np.atleast_2d(layer.delta).T, output).T * rate

            # 更新偏置
            # 对于偏置矩阵，系数为1，所以delta就是它的梯度矩阵
            layer.bias -= layer.delta * rate

    def score(self, X, Y):
        y_pred = []
        y_true = []
        total_loss = []
        for x, y in zip(X, Y):
            output = self.forward_propagation(x)
            output = np.exp(output - output.max(axis=-1, keepdims=True))
            output = output / output.sum(axis=-1, keepdims=True)

            y_pred.append(np.argmax(output))
            y_true.append(np.argmax(y))
            loss = -np.sum(y * np.log(output))
            total_loss.append(loss)
        return accuracy_score(y_pred, y_true), np.mean(total_loss)

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


def tanh(x):  # 定義tanh函數
    return (1.0 - np.exp(-x))/(1.0 + np.exp(-x))


def tanh_derivative(x):  # 定義tanh導數
    return (1+tanh(x))*(1-tanh(x))


class NeuralNetwork():  # 定義列別NeuralNetwork
    # 定義啟動函數
    # net_arch為整數，表示每層中神經元的數量
    def __init__(self, net_arch):
        self.activity = tanh
        self.activity_derivative = tanh_derivative
        self.layers = len(net_arch)
        self.steps_per_epoch = 1000
        self.arch = net_arch

        self.weights = []
        # 權重範圍為-1~1
        for layer in range(self.layers-1):
            w = 2*np.random.rand(net_arch[layer]+1, net_arch[layer]-1)
            self.weights.append(w)  # 初始化權重

    def fit(self, data, labels, learning_rate=0.1, epochs=100):  # 訓練類神經網路
        # 向輸入添加偏移單元
        ones = np.ones((1, data.shape[0]))
        Z = np.concatenate((ones.T, data), axis=1)
        training = epochs*self.steps_per_epoch
        for k in range(training):
            if k % self.steps_per_epoch == 0:
                print('epochs:{}'.format(k/self.steps_per_epoch))
                for s in data:
                    print(s, nn.predict(s))

        # 設定前饋
        sample = np.random.randint(data.shape[0])
        y = [Z[sample]]
        print(y)
        for i in range(len(self.weights)-1):
            activation = np.dot(y[i], self.weights[i])
            activity = self.activity(activation)
            activity = np.concatenate(np.ones(1), np.array(activity))
            y.append(activity)

        # last layer
        activation = np.dot(y[-1], self.weights[-1])
        activity = self.activity(activation)
        y.append(activity)

		# 反傳遞
        error = labels[sample]-y[-1]
        delta_vec = [error * self.activity_derivative(y[-1])]
        #we need to begin from the back 
        #from the next to last layer
        for i in range(self.layers-2, 0, -1):
            error = delta_vec[-1].dot(self.weights[i][1:].T)
            error = error * self.activity_derivative(y[i][1:])
            delta_vec.append(error)

        delta_vec.reverse()
        for i in range(len(self.weights)):
            layer = y[i].reshape(1, nn.arch[i]+1)
            delta = delta_vec.reshape(1, nn.arch[i+1])
            self.weights[i] += learning_rate*layer.T.dot(delta)

    def predict(self, x):
        val = np.concatenate((np.ones(1).T, np.array(x)))
        for i in range(0, len(self.weights)):
            val = self.activity(np.dot(val, self.weights[i]))
            val = np.concatenate((np.ones(1).T, np.array(val)))
        return val[1]


if __name__ == '__main__':
    np.random.seed(0)
    nn = NeuralNetwork([2, 2, 1])
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    nn.fit(x, y, epochs=10)

    print("Final prediction")
    for s in x:
        print(s, nn.predict(s))

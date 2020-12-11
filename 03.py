import numpy as np
import mnist


def ReLU(x):
    return np.maximum(0, x)


def ReLU_derivative(x):
    return 1 if x > 0 else 0


mndata = mnist.MNIST('CNNs/bin')
mndata.gz = True


train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

print("數據:")
print(len(train_images[1]))


class NeuralNetwork:
    def __init__(self, net_arch):
        self.activity = ReLU
        self.activity_derivative = ReLU_derivative
        self.layers = len(net_arch)
        self.steps_per_epoch = 1000
        self.arch = net_arch

        self.weights = []

        for layer in range(self.layers-1):
            w = 2*np.random.rand(net_arch[layer]+1, net_arch[layer]-1)
            self.weights.append(w)

    def fit(self, data, labels, learning_rate=0.1, epochs=100):
        ones = np.ones((1, data.shape([0])))
        z = np.concatenate((ones.T, data), axis=1)
        training = epochs*self.steps_per_epoch
        for k in range(training):
            if k % self.steps_per_epoch == 0:
                print('epochs:{}'.format(k/self.steps_per_epoch))
                for s in data:
                    print(s, nn.predict(s))

        sample = np.random.randint(data.shape([0]))
        y = [z[sample]]
        for i in range(len(self.weights)-1):
            activation = np.dot(y[i], self.weights[i])
            activity = self.activity(activation)
            activity = np.concatenate((np.ones(1),
                                       np.array(activity)))
            y.append(activity)
        activation = np.dot(y[-1], self.weights[-1])
        activity = self.activity(activation)
        y.append(activity)

        error = labels[]

if __name__ == '__main__':
    np.random.seed(0)
    nn = NeuralNetwork([2, 4, 1])

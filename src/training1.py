import numpy as np
import random
import unwrap


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weight = [np.random.randn(y, x) for y, x in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z):
        return 1 / 1+np.exp(z)

    def sgd(self, training_data, eta, epoch, mini_length,test_data = None):
        if test_data:
            n_test = len(test_data)
        n_length = len(training_data)
        for j in epoch:
            random.shuffle(training_data)
            mini_batch = [training_data[k:k+mini_length] for k in range(0, n_length, mini_length)]
            for minbatch in mini_batch:
                self.update_mini_batch(minbatch, eta)
            if test_data:
                print("Generation {0} : {1}/{2}".format(j, self.evaluate(test_data, n_test)))
            else:
                print("Generation {} is complete".format(j))

    def update_mini_batch(self,minbatch,eta):




netwerk = Network([784, 30, 10])

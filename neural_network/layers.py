import numpy as np


class Layer:

    def __init__(self, nodes):
        self.nodes = nodes
        self.layer_input = None
        self.layer_output = None
        self.weights = None
        self.bias = None
        self.level = None

    def forward_pass(self, layer_input):
        """
        """
        z = np.dot(layer_input, self.weights) + self.bias

        output = self.activation_function(z)

        self.layer_input = layer_input
        self.layer_output = output

        return output

    def activation_function(self, z):
        """
        """
        raise NotImplementedError

    def activation_derivative(self):
        """
        """
        raise NotImplementedError

    def update_parameters(self,
                          learning_rate,
                          delta_weight,
                          delta_bias,
                          n_records):
        """
        Update weights and bias on gradient descent step

        delta_weight_matrix: change in weights in each hidden layer
        delta_bias_matrix: change in bias in each hidden layer
        n_records: number of records
        """

        self.weights += learning_rate * delta_weight / n_records
        self.bias += learning_rate * delta_bias / n_records


class SigmoidLayer(Layer):

    def activation_function(self, z):
        """
        """
        return 1 / (1 + np.exp(-z))

    def activation_derivative(self):
        """
        """
        return self.layer_output * (1 - self.layer_output)


class ReLULayer(Layer):

    def activation_function(self, z):
        """
        """
        return np.maximum(0, z)

    def activation_derivative(self):
        """
        """
        return np.heaviside(self.layer_output, 0)


class TanhLayer(Layer):

    def activation_function(self, z):
        """

        :param z:
        :return:
        """
        return np.tanh(z)

    def activation_derivative(self):
        """

        :return:
        """
        return 1 - np.tanh(self.layer_output) ** 2


class LeakyReLULayer(Layer):

    @staticmethod
    def helper(x, y=None):
        """

        :param x:
        :param y:
        :return:
        """
        if x < 0:
            return 0.01
        else:
            if y:
                return y
        return x

    def activation_function(self, z):
        """

        :param z:
        :return:
        """

        np_helper = np.vectorize(LeakyReLULayer.helper)
        return np_helper(z)

    def activation_derivative(self):
        """

        :return:
        """

        np_helper = np.vectorize(LeakyReLULayer.helper)

        return np_helper(self.layer_output, 1)

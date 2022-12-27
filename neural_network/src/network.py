import numpy as np

from neural_network.src.layers import (
    LeakyReLULayer,
    ReLULayer,
    SigmoidLayer,
    TanhLayer
)

from neural_network.src.loss_function import (
    binary_cross_entropy,
    binary_cross_entropy_prime,
    cross_entropy,
    cross_entropy_prime,
    mean_squared_error,
    mean_squared_error_prime

)


class NeuralNetwork:

    def __init__(self,
                 input_nodes: int,
                 layers: list,
                 learning_rate: float,
                 loss_function="BCE"
                 ):
        self.input_nodes = input_nodes
        self.layers = layers
        self.learning_rate = learning_rate
        self.network = []
        self.architecture = []
        self.loss_function = loss_function

    def create_network(self):
        """
        Creates the layers of the neural network
        """
        network_input = self.input_nodes
        np.random.seed(99)
        for idx, layer in enumerate(self.layers):

            nodes = layer[0]
            activation_function = layer[1]

            if activation_function == "ReLU":
                network_layer = ReLULayer(nodes)
            elif activation_function == "tanh":
                network_layer = TanhLayer(nodes)
            elif activation_function == "LeakyReLU":
                network_layer = LeakyReLULayer(nodes)
            else:
                network_layer = SigmoidLayer(nodes)

            network_layer.level = idx

            network_layer.weights = np.random.uniform(-1, 1, (network_input, nodes))

            network_layer.bias = np.random.normal(0, 1, (1, nodes))
            self.network.append(network_layer)

            self.architecture.append({
                "input_dim": network_input,
                "output_dim": nodes,
                "activation": activation_function
            })

            network_input = nodes

        return self

    def feed_forward(self, network_input):
        """
        Forward pass through the network

        :param network_input: input data to the network

        :return: final output of the network
        :rtype: tuple of nD arrays
        """
        inputs = network_input
        outputs = 0
        for layer in self.network:
            outputs = layer.forward_pass(inputs)

            inputs = outputs

        return outputs

    @staticmethod
    def calculate_loss(actual, predicted, loss_function):
        """
        Check for the loss function to be used
        Calculate and return the loss_score and derivate of loss_score

        :param actual: Actual output given in the dataset
        :param predicted: Predicted output using feedforward
        :param loss_function: Loss function to calculate error of the network

        :return: loss_score - Error of the whole network
        :return: loss_derivative - Derivative of the loss score
        :rtype: tuple
        """
        if loss_function == "MSE":
            loss_score = mean_squared_error(predicted, actual)
            loss_derivative = mean_squared_error_prime(predicted, actual)

            return loss_score, loss_derivative

        if loss_function == "BCE":
            loss_score = binary_cross_entropy(predicted, actual)
            loss_derivative = binary_cross_entropy_prime(predicted, actual)

            return loss_score, loss_derivative

        if loss_function == "CE":
            loss_score = cross_entropy(predicted, actual)
            loss_derivative = cross_entropy_prime(predicted, actual)

            return loss_score, loss_derivative

    def backpropagation(self,
                        delta_weight_matrix,
                        delta_bias_matrix,
                        loss_derivative
                        ):
        """
        Backpropagation using log loss error function

        :param delta_weight_matrix: change in weights for all layers
        :param delta_bias_matrix: change in bias for all layers
        :param loss_derivative: derivative of the loss score

        :return: delta_weight_matrix
        :return: delta_bias_matrix
        :rtype: tuple of nD arrays
        """
        # d_loss is gradient of the loss on the output
        d_out = loss_derivative

        # start from the output layer
        for i, layer in reversed(list(enumerate(self.network))):
            # chain d_out with gradient of output to activation
            da = d_out * layer.activation_derivative()

            # chain da with gradient of output to weight
            dw = np.dot(layer.layer_input.T, da)

            delta_weight_matrix[i] = dw
            # chain da with gradient of output to bias
            db = np.sum(da, axis=0, keepdims=True)

            delta_bias_matrix[i] = db

            # d_out for the next layer is scaled by the weights in the curr layer
            d_out = np.dot(d_out, layer.weights.T)

        return delta_weight_matrix, delta_bias_matrix

    def update_parameters(self,
                          delta_weight_matrix,
                          delta_bias_matrix,
                          n_records
                          ):
        """
        Update weights and bias on gradient descent step

        :param delta_weight_matrix: change in weights for all layers
        :param delta_bias_matrix: change in bias in for all layers
        :param n_records: number of records
        """
        for i, layer in enumerate(self.network):
            layer.update_parameters(self.learning_rate,
                                    delta_weight_matrix[i],
                                    delta_bias_matrix[i],
                                    n_records)

    def train(self, network_input, output):
        """
        Train the network on batch of input features and targets.

        :param network_input: input data to the network
        :param output: target values

        :return: network output
        :return: loss
        :rtype: tuple
        """
        n_records = network_input.shape[0]

        # We need to record this, because we can only make the updates after
        # a full back prop
        delta_weight_matrix = []  # change in weights for all layers
        delta_bias_matrix = []  # change in bias for all layers

        # add dw and db for each layer into matrices
        # delta_weight_matrix and delta_bias_matrix
        for layer in self.network:
            delta_weight_matrix.append(layer.weights)
            delta_bias_matrix.append(layer.bias)

        network_output = self.feed_forward(network_input)

        loss_score, loss_derivative = self.calculate_loss(output,
                                                          network_output,
                                                          self.loss_function)

        delta_weight_matrix, delta_bias_matrix = self.backpropagation(delta_weight_matrix,
                                                                      delta_bias_matrix,
                                                                      loss_derivative
                                                                      )

        self.update_parameters(delta_weight_matrix, delta_bias_matrix, n_records)

        return network_output, loss_score

    def validate(self, validation_input, validation_output):
        """
        Performs validation during training
        :param validation_input: input data
        :param validation_output: target values

        :return: loss
        :rtype: 1D array
        """

        network_output = self.feed_forward(validation_input)

        if self.loss_function == "MSE":
            loss_score = mean_squared_error(network_output, validation_output)

        elif self.loss_function == "BCE":
            loss_score = binary_cross_entropy(network_output, validation_output)

        elif self.loss_function == "CE":
            loss_score = cross_entropy(network_output, validation_output)

        return loss_score

__author__ = 'vapaspen'

import numpy as np


class RNN:
    """
    Class to make a test RNN
    """

    def __init__(self, layer_definitions=None):
        self.layer_definitions = layer_definitions
        self.Layers = {}


class RNN_Layer:
    """
    Basic fully connected RNN Layer
    """

    def __init__(self, X_count, H_count):
        self.Layer = {}

        self.Layer["X_count"] = X_count
        self.Layer["H_count"] = H_count

        self.Layer["Last_input"] = np.zeros((self.Layer["X_count"],), dtype=float)

        self.Layer["nodes"] = {}

        self.Layer["nodes"]["neurons"] = np.zeros((self.Layer["H_count"],), dtype=float)
        self.Layer["nodes"]["neurons_delta"] = np.zeros_like(self.Layer["nodes"]["neurons"])

        self.Layer["nodes"]["bias"] = np.zeros((self.Layer["H_count"],), dtype=float)
        self.Layer["nodes"]["bias_delta"] = np.zeros_like(self.Layer["nodes"]["bias"])

        self.Layer["nodes"]["hidden_weights"] = np.random.random((self.Layer["H_count"], self.Layer["H_count"]))*0.1
        self.Layer["nodes"]["hidden_weights_delta"] = np.zeros_like(self.Layer["nodes"]["hidden_weights"])

        self.Layer["nodes"]["input_weights"] = np.random.random((self.Layer["X_count"], self.Layer["H_count"]))*0.1
        self.Layer["nodes"]["input_weights_delta"] = np.zeros_like(self.Layer["nodes"]["input_weights"])

    def feed_foward(self, input_nodes):
        """
        The basic feed forward Process for this layer.

        :param input_nodes:
            One dimensional Numpy array with the same length as the configured X size
        :return:
            Activated One dimensional numpy array

        :raises:
            Exception when input length doesnt match input settings.
        """
        if not len(input_nodes) == self.Layer["X_count"]:
            raise Exception("Input given not the same shape as Layer settings.")

        self.Layer["Last_input"] = input_nodes
        self.Layer["nodes"]["neurons"] = np.tanh(np.dot(self.Layer["nodes"]["neurons"],  self.Layer["nodes"]["hidden_weights"]) + np.dot(input_nodes, self.Layer["nodes"]["input_weights"]) + self.Layer["nodes"]["bias"])
        return self.Layer["nodes"]["neurons"]

    def reset_bias(self):
        """
        Function to reset all of the Gradients for this layer
        :return: void
        """
        self.Layer["nodes"]["neurons_delta"] = np.zeros_like(self.Layer["nodes"]["neurons"])
        self.Layer["nodes"]["bias_delta"] = np.zeros_like(self.Layer["nodes"]["bias"])
        self.Layer["nodes"]["input_weights_delta"] = np.zeros_like(self.Layer["nodes"]["input_weights"])
        self.Layer["nodes"]["hidden_weights_delta"] = np.zeros_like(self.Layer["nodes"]["hidden_weights"])


class ANN_Layer:
    """
    Basic fully connected ANN Layer
    """

    def __init__(self, X_count, H_count):
        self.Layer = {}

        self.Layer["X_count"] = X_count
        self.Layer["H_count"] = H_count

        self.Layer["Last_input"] = np.zeros((self.Layer["X_count"],), dtype=float)

        self.Layer["nodes"] = {}

        self.Layer["nodes"]["neurons"] = np.zeros((self.Layer["H_count"],), dtype=float)
        self.Layer["nodes"]["neurons_delta"] = np.zeros_like(self.Layer["nodes"]["neurons"])

        self.Layer["nodes"]["bias"] = np.zeros((self.Layer["H_count"],), dtype=float)
        self.Layer["nodes"]["bias_delta"] = np.zeros_like(self.Layer["nodes"]["bias"])

        self.Layer["nodes"]["input_weights"] = np.random.random((self.Layer["X_count"], self.Layer["H_count"]))*0.1
        self.Layer["nodes"]["input_weights_delta"] = np.zeros_like(self.Layer["nodes"]["input_weights"])

    def feed_foward(self, input_nodes):
        """
        The basic feed forward Process for this layer.

        :param input_nodes:
            One dimensional Numpy array with the same length as the configured X size

        :return:
            Activated One dimensional numpy array

        :raises:
            Exception when input length doesnt match input settings.
        """

        if not len(input_nodes) == self.Layer["X_count"]:
            raise Exception("Input given not the same shape as Layer settings.")

        self.Layer["Last_input"] = input_nodes
        self.Layer["nodes"]["neurons"] = np.tanh(np.dot(input_nodes, self.Layer["nodes"]["input_weights"]) + self.Layer["nodes"]["bias"])
        return self.Layer["nodes"]["neurons"]

    def reset_bias(self):
        """
        Function to reset all of the Gradients for this layer
        :return: void
        """
        self.Layer["nodes"]["neurons_delta"] = np.zeros_like(self.Layer["nodes"]["neurons"])
        self.Layer["nodes"]["bias_delta"] = np.zeros_like(self.Layer["nodes"]["bias"])
        self.Layer["nodes"]["input_weights_delta"] = np.zeros_like(self.Layer["nodes"]["input_weights"])

    def backpropagate(self, test_error, learning_rate=0.01):
        """
        :param test_error:
            One dimensional Numpy array that represents the Gradients of this layer Post activation.
        :return:
            One dimensional Numpy array that represents the Gradients of this layer Post activation of the Next Layer
        """
        self.reset_bias()
        self.Layer["nodes"]["neurons_delta"] = test_error
        self.Layer["nodes"]["bias_delta"] += (1 - self.Layer["nodes"]["neurons"] * self.Layer["nodes"]["neurons"]) * test_error
        self.Layer["nodes"]["input_weights_delta"] += self.Layer["nodes"]["bias_delta"] * np.array([self.Layer["Last_input"]]).T

        #Update all Layer Parameters at once.
        for param, delta_param in zip([self.Layer["nodes"]["bias"], self.Layer["nodes"]["input_weights"]],
                                      [self.Layer["nodes"]["bias_delta"], self.Layer["nodes"]["input_weights_delta"]]):
            np.clip(delta_param, -10, 10, out=delta_param)
            param += -learning_rate * delta_param

        return np.dot(self.Layer["nodes"]["bias_delta"], self.Layer["nodes"]["input_weights"].T)





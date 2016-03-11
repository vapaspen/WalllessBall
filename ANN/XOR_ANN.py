__author__ = 'vapaspen'

"""
    test XOR ANN used as a Demo for proof of understanding of the back backpropagational algorithms.

"""

import numpy as np
import random

class FNN:

    def __init__(self, num_input=0, num_hidden=0, num_out=0, learn_rate=.1):
        self.learning_rate = learn_rate
        self.error = 0

        #Neuron Count store
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_out = num_out

         # initialize Node parts: weights and Bias and and node states
        self.hidden_weights = np.random.random((self.num_hidden, self.num_input))*0.01
        self.output_weights = np.random.random((self.num_out, self.num_hidden))*0.01
        #print("self.hidden_weights" + str(self.hidden_weights))

        self.hidden_layer = np.zeros((self.num_hidden,), dtype=float)
        self.output_layer = np.zeros((self.num_out,), dtype=float)

        self.hidden_bias = np.zeros((self.num_hidden,), dtype=float)
        self.output_bias = np.zeros((self.num_out,), dtype=float)
        #print("self.hidden_bias: " + str(self.hidden_bias))

        #Expermintal
        self.last_delta_hidden_bias = np.zeros_like(self.hidden_bias)
        self.last_delta_output_bias = np.zeros_like(self.output_bias)
        self.last_delta_hidden_weights = np.zeros_like(self.hidden_weights)
        self.last_delta_output_weights = np.zeros_like(self.output_weights)

        self.hidden_bias.fill(1)
        self.output_bias.fill(0)


    def feed_foward(self, input_layer):
        self.hidden_layer = np.dot(self.hidden_weights, input_layer) + self.hidden_bias
        #print("self.hidden_layer: " + str(self.hidden_layer))
        self.hidden_layer = np.tanh(self.hidden_layer)
        #print("self.hidden_weights: " + str(self.hidden_weights))
        self.output_layer = np.dot(self.output_weights, self.hidden_layer) + self.output_bias
        #self.output_layer = np.tanh(self.output_layer)
        self.output_layer = np.tanh(self.output_layer)

        #print("self.output_layer: " + str(self.output_layer))
        return self.output_layer

    def train_on_data(self, input_layer, target):
        delta_hidden_bias = np.zeros_like(self.hidden_bias)
        delta_output_bias = np.zeros_like(self.output_bias)
        delta_hidden_weights = np.zeros_like(self.hidden_weights)
        delta_output_weights = np.zeros_like(self.output_weights)

        delta_output_layer = np.zeros_like(self.output_layer)
        delta_hidden_layer = np.zeros_like(self.hidden_layer)

        input_as_array = np.zeros_like(input_layer)
        input_as_array += input_layer

        #print("input_layer: " + str(input_layer))
        current_pass = self.feed_foward(input_layer)

        #print("current_pass: " + str(current_pass))

        """
            Cost Function:
        """
        pass_difference = (target - current_pass) * -1
        self.error = pass_difference

        #self.smoth_error = self.smoth_error * 0.999 + pass_difference * 0.001

        #print("self.error: " + str(self.error))

        delta_output_layer += self.output_layer * self.error
        #print("delta_output_layer: " + str(delta_output_layer))

        delta_output_bias += (1 - self.output_layer * self.output_layer) * delta_output_layer
        #delta_output_bias += (2 * self.output_layer) * delta_output_layer
        #print("self.output_bias: " + str(self.output_bias))
        #print("delta_output_bias: " + str(delta_output_bias))

        delta_output_weights += self.hidden_layer * delta_output_bias
        #print("self.output_weights: " + str(self.output_weights))
        #print("delta_output_weights: " + str(delta_output_weights))


        delta_hidden_layer += np.dot(self.output_weights.T, delta_output_bias)
        #print("delta_hidden_layer: " + str(delta_hidden_layer))
        #print("self.hidden_layer: " + str(self.hidden_layer))


        delta_hidden_bias += (1 - self.hidden_layer * self.hidden_layer) * delta_hidden_layer
        #print("delta_hidden_bias " + str(delta_hidden_bias))
        #print("self.hidden_bias " + str(self.hidden_bias))

        delta_hidden_weights += np.dot(delta_hidden_bias, input_as_array)
        #delta_hidden_weights += delta_hidden_bias * input_as_array

        #print("input_as_array.T " + str(input_as_array))
        #print("Befor: delta_hidden_weights: " + str(delta_hidden_weights))
        #print("Befor: delta_hidden_weights: " + str(delta_hidden_weights))

        for param, delta_param, mem_param in zip([self.output_bias, self.output_weights, self.hidden_bias, self.hidden_weights],
                                      [delta_output_bias, delta_output_weights, delta_hidden_bias, delta_hidden_weights],
                                        [self.last_delta_output_bias, self.last_delta_output_weights, self.last_delta_hidden_bias, self.last_delta_hidden_weights]):



            np.clip(delta_param, -10, 10, out=delta_param)

            param += -self.learning_rate * delta_param








if __name__ == 'main':
    """
        Run Module
    """



#define Data
zero = 1
one = 0
xor_data = [[zero, zero], [zero, one], [one, zero], [one, one]]
xor_data_train = [one, zero, zero, one]

NN = FNN(num_input=2, num_hidden=2, num_out=1)

#NN.train_on_data(xor_data[1], xor_data_train[1])

test_error = .0001

n = 0
run = True
while run:
    #for i in range(4):
    i = random.randrange(4)
    NN.train_on_data(xor_data[i], xor_data_train[i])
    test_error = (NN.error + test_error) /2
    if n % 500 == 0:
        print("Trial: " + str(n) + " Index: " + str(i) + " Error: " + str(test_error) + " Result: " + str(NN.output_layer) + " Expected: " + str(xor_data_train[i]))
        #print("self.delta_del_delta_hidden_bias: " + str(np.mean(NN.delta_del_delta_hidden_bias)))
        #print("self.delta_del_hidden_bias: " + str(NN.delta_del_hidden_bias))
        if i == 1:
            #print("Index: " + str(i) + " Error: " + str(NN.error) + " Result: " + str(NN.output_layer) + " Expected: " + str(xor_data_train[i]))
            pass

        pass_test = 0

        pass_test += NN.feed_foward(xor_data[0])
        pass_test -= NN.feed_foward(xor_data[1])
        pass_test -= NN.feed_foward(xor_data[2])
        pass_test += NN.feed_foward(xor_data[3])
        pass_test += NN.feed_foward(xor_data[0])
        pass_test -= NN.feed_foward(xor_data[1])
        pass_test -= NN.feed_foward(xor_data[2])
        pass_test += NN.feed_foward(xor_data[3])

        if pass_test > 3.90 and pass_test < 4.01:
            print("Pass test" + str(pass_test))
            run = False
    #print(" ")


    n = n + 1
    if n > 100000:
        run = False
    #print("--------")
    #print(" ")

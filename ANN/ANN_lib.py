__author__ = 'vapaspen'

#import numpy as np
import random
import math

#activation Functions:
def tanh(node_list, prime=False):
    output_list = []
    if not prime:
        for node in node_list:
            output_list.append(math.tanh(node))
    else:
        for node in node_list:
            output_list.append(1 - node * node)
    return output_list

def zeros_list(num):
    list_of_zeros = [1.0] * num
    return list_of_zeros

def zeros_list_like(template_list):
    if type(template_list[0]) == list:
        list_of_zeros = [[0.0] * len(template_list[0])] * len(template_list)
    else:
        list_of_zeros = [0.0] * len(template_list)
    return list_of_zeros

def get_random(modu=1):
    return random.random()*modu

def add_list(list_one, list_two):
    result = []
    for count in range(len(list_one)):
        result.append(list_one[count] + list_two[count])
    return result

def dot_list(number_list_left, number_list_right):
    """
    Returns the dot product of two lists. Will Process lists Across a second dimension.
    :param number_list_left: Single dimension list
    :param number_list_right: one or dimensional list
    :return: a scalar or list of scalars.
    """

    if type(number_list_right[0]) == list:
        sum_of_lists = []
        for count in range(len(number_list_right)):
            if not len(number_list_left) == len(number_list_right[count]):
                raise Exception("Lists must be the same size")
            sum_of_lists_sub = 0
            for subcount in range(len(number_list_right[count])):
                sum_of_lists_sub += number_list_left[subcount] * number_list_right[count][subcount]
            sum_of_lists.append(sum_of_lists_sub)

    else:
        if not len(number_list_left) == len(number_list_right):
            raise Exception("Lists must be the same size")
        sum_of_lists = 0
        for count in range(len(number_list_left)):
            sum_of_lists += number_list_left[count] * number_list_right[count]
    return sum_of_lists

class Hidden_Layer:
    """Basic Hidden Layer for a Neural Network.

    Used to store the layer state, definition, feed forward and packaging Methods.

    Keyword Args:
        x_count: Number of Nodes that will be sending Inputs to this layer
        h_count: Number of Nodes in this hidden layer.
        is_recurrent: sets the layer to act in a recurrent way. Default is False
        layer: when used it will make the new layer by unpacking a provided packaged layer as dict
        activation: The activation function used for this layer. Is type function that is a reference to one of the predefined functions that includes the derivative.

    """


    def __init__(self, x_count, h_count, is_recurrent=False, layer=None):

        if not layer:
            self.layer = {}

            #Layer Paramiters
            self.layer['param'] = {}
            self.layer['param']['num_inputs'] = x_count
            self.layer['param']['num_neurons'] = h_count
            self.layer['param']['is_recurrent'] = is_recurrent

            #Layer Node storage
            self.layer['nodes'] = {}

            self.layer['nodes']['node_states'] = zeros_list(self.layer['param']['num_neurons'])
            self.layer['nodes']['bias'] = zeros_list(self.layer['param']['num_neurons'])

            self.layer['nodes']['node_delta'] = zeros_list_like(self.layer['nodes']['node_states'])
            self.layer['nodes']['bias_delta'] = zeros_list_like(self.layer['nodes']['bias'])

            self.layer['nodes']['weights'] = []
            for node in range(self.layer['param']['num_neurons']):
                stub_sub = []
                for i in range(self.layer['param']['num_inputs']):
                    stub_sub.append(get_random(modu=0.01))
                self.layer['nodes']['weights'].append(stub_sub)

            self.layer['nodes']['delta_weights'] = zeros_list_like(self.layer['nodes']['weights'])


            if is_recurrent:
                self.layer['nodes']['hidden_weights'] = []

                for node in range(self.layer['param']['num_neurons']):
                    stub_sub = []
                    for i in range(self.layer['param']['num_neurons']):
                        stub_sub.append(get_random(modu=0.01))
                    self.layer['nodes']['hidden_weights'].append(stub_sub)

                self.layer['nodes']['delta_hidden_weights'] = zeros_list_like(self.layer['nodes']['hidden_weights'])

    def feed_forward(self, input_layer):
        W = dot_list(input_layer, self.layer['nodes']['weights'])


        WB = add_list(W, self.layer['nodes']['bias'])

        if self.layer['param']['is_recurrent']:
            H = dot_list(self.layer['nodes']['node_states'], self.layer['nodes']['hidden_weights'])
            WB = add_list(WB, H)

        WBA = tanh(WB)
        self.layer['nodes']['node_states'] = WBA
        return WBA

Test = Hidden_Layer(2,4, is_recurrent=True)
print(Test.feed_forward([2, 3]))
print(Test.feed_forward([2, 3]))

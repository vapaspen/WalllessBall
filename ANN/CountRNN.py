__author__ = 'vapaspen'


import RNN_lib as rl
import random
import numpy as np

class Experimental_ANN:

    def __init__(self, X, H, O, learning_rate=0.1):
        self.H = rl.ANN_Layer(X, H, is_recurrent=True)
        self.OL = rl.ANN_Layer(H, O, is_recurrent=True)
        self.OR = rl.ANN_Layer(H, O, is_recurrent=True)

        self.learning_rate = learning_rate
        self.errorL = 0
        self.errorR = 0

    def feed_foward(self, input_data):
        H = self.H.feed_foward(input_data)

        OR = self.OR.feed_foward(H)



        return [self.OL.feed_foward(H),OR]

    def train_on_data(self, data, targetL, targetR):
        this_pass = self.feed_foward(data)
        self.errorL = this_pass[0] - targetL
        self.errorR = this_pass[1] - targetR

        h_error = self.OR.backpropagate(self.errorR, learning_rate=self.learning_rate) + self.OL.backpropagate(self.errorL, learning_rate=self.learning_rate)
        self.H.backpropagate(h_error, learning_rate=self.learning_rate)



#define Data
zero = 1
one = 0
xor_data = [[zero, zero], [zero, one], [one, zero], [one, one]]
xor_data_trainL = [one, zero, zero, one]
xor_data_trainR = [zero, one, one, zero]

NN = Experimental_ANN(2, 3, 1)

test_error = .0001

n = 0
run = True
while run:
    #for i in range(4):
    i = random.randrange(4)
    NN.train_on_data(xor_data[i], xor_data_trainL[i], xor_data_trainR[i])
    test_error = (NN.errorL  + NN.errorR + test_error) /2
    if n % 1000 == 0:
        print("Trial: " + str(n) + " Index: " + str(i) + " Error: " + str(test_error) + " Result: Left: " + str(NN.OL.Layer["nodes"]["neurons"]) + " Right:" + str(NN.OR.Layer["nodes"]["neurons"]) + " Expected: " + str(xor_data_trainL[i]) + ", " + str(xor_data_trainR[i]))

        """
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

"""
    n = n + 1
    if n > 100000:
        run = False


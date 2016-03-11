__author__ = 'vapaspen'


from sense_hat import SenseHat
from time import sleep

"""
Basic frame componets
"""
S = [0, 50, 180] #shirt
P = [50, 50, 50] #Pants
Sh = [180, 0, 180] #shoes
Sk = [255, 160, 60] #Skin
H = [180, 180, 0] #Hair
O = [0, 0, 0]  # off

man_stand = [
O, O, O, O, H, O, O, O,
O, O, O, Sk, Sk, Sk, O, O,
O, O, O, O, Sk, O, O, O,
O, O, Sk, S, S, S, Sk, O,
O, O, O, O, S, O, O, O,
O, O, O, O, P, O, O, O,
O, O, O, P, O, P, O, O,
O, O, Sh, O, O, O, Sh, O
]

man_armsUp_legsIn = [
O, O, O, O, H, O, O, O,
O, O, O, Sk, Sk, Sk, O, O,
O, O, Sk, O, Sk, O, Sk, O,
O, O, O, S, S, S, O, O,
O, O, O, O, S, O, O, O,
O, O, O, O, P, O, O, O,
O, O, O, P, O, P, O, O,
O, O, O, Sh, O, Sh, O, O
]

man_armsDown_Legsout = [
O, O, O, O, H, O, O, O,
O, O, O, Sk, Sk, Sk, O, O,
O, O, O, O, Sk, O, O, O,
O, O, O, S, S, S, O, O,
O, O, Sk, O, S, O, Sk, O,
O, O, O, O, P, O, O, O,
O, O, O, P, O, P, O, O,
O, O, Sh, O, O, O, Sh, O
]


class Man:
    def __init__(self):
        self.current_frame = man_stand
        self.sense = SenseHat()
        self.sense.rotation = 180
        self.sense.set_pixels(self.current_frame)
        sleep(.5)


    def jumping_jack(self):
        """
        Displays the frams needed to do one jumping Jack

        :return:
        """
        sequance = [man_armsUp_legsIn, man_stand, man_armsDown_Legsout, man_stand]

        for frame in sequance:
            self.current_frame = frame
            self.sense.set_pixels(frame)
            sleep(.5)

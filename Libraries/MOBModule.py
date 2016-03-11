__author__ = 'vapaspen'

import math

class MObj(object):
    """
        Definition of the Mobile Object in game.

        Will default to a postion of x: 0, y: 0, x_velocity = 0, y_velocity: 0, mass: 100, mu: .001

        Args:
            x: starting X position
            y: Starting Y position
            x_velocity: Starting velocity in the X direction
            y_velocity: Starting velocity in the Y direction
            mass: The Mass of the Object
            mu: The frictional coefficient of the surface of this object.

        Methods:


        Raises:
    """
    def __init__(self, x=0, y=0, x_velocity=0, y_velocity=0, mass=100, mu=.001):
        self.x = x
        self.y = y
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.mass = mass
        self.mu = mu



class Simulation(object):
    """
        Parent Object for the game Simulation.

        Stores all objects in teh simulation. contains the methods for setting the position of those objects.

        Args:
            Gravity: gravity in the simulation
            friction of on the base
    """
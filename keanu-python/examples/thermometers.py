import keanu as kn
import numpy as np

class Thermometers():

    def __init__(self):
        pass

    def model(self):

        with kn.Model() as m:
        	m.temperature = kn.Uniform(0., 100.)
        	m.thermometer_one = kn.Gaussian(m.temperature, 1.0)
        	m.thermometer_two = kn.Gaussian(m.temperature, 1.0)

        return m
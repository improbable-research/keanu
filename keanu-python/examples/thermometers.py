import keanu as kn

def model():

    with kn.Model() as m:
    	m.temperature = kn.Uniform(0., 100.)
    	m.thermometer_one = kn.Gaussian(m.temperature, 1.0)
    	m.thermometer_two = kn.Gaussian(m.temperature, 1.0)

    return m
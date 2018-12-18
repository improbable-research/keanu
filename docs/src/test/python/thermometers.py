from keanu import BayesNet, KeanuRandom, Model
from keanu.vertex import Gamma, Exponential, Cauchy, Gaussian, Uniform
from keanu.algorithm import GradientOptimizer

def thermometers_example():
    # %%SNIPPET_START%% PythonTwoThermometers
    with Model() as m:
        m.temperature = Uniform(20., 30.)
        m.first_thermometer = Gaussian(m.temperature, 2.5)
        m.second_thermometer = Gaussian(m.temperature, 5.)

    m.first_thermometer.observe(25.)
    m.second_thermometer.observe(30.)

    bayes_net = m.to_bayes_net()
    optimizer = GradientOptimizer(bayes_net)
    optimizer.max_a_posteriori()

    calculated_temperature = m.temperature.get_value()
    print(calculated_temperature)
    # %%SNIPPET_END%% PythonTwoThermometers

thermometers_example()
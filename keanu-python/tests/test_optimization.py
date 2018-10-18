import keanu as kn
from examples import Thermometers

def test_gradient_op_bayes_net():
    thermometers = Thermometers()
    model = thermometers.model()
    net = kn.BayesNet(model.temperature.getConnectedGraph())
    gradient_optimizer = kn.GradientOptimizer(net)
    pass


def test_gradient_op_vertex():
    thermometers = Thermometers()
    model = thermometers.model()
    gradient_optimizer = kn.GradientOptimizer(model.temperature)
    pass


def test_gradient_can_set_builder_properties():
    thermometers = Thermometers()
    model = thermometers.model()
    gradient_optimizer = kn.GradientOptimizer(model.temperature, 5000, 0.001, 0.002)
    pass


def test_thermometers_map_gradient():
    thermometers = Thermometers()
    model = thermometers.model()

    model.thermometer_one.observe(22.0)
    model.thermometer_two.observe(20.0)

    net = kn.BayesNet(model.temperature.getConnectedGraph())
    gradient_optimizer = kn.GradientOptimizer(net)

    logProb = gradient_optimizer.max_a_posteriori()

    assert logProb < 0.

    temperature = model.temperature.getValue().scalar()

    assert temperature > 20.995 and temperature < 21.005


def test_thermometers_max_likelihood_gradient():
    thermometers = Thermometers()
    model = thermometers.model()

    model.thermometer_one.observe(22.0)
    model.thermometer_two.observe(20.0)

    net = kn.BayesNet(model.temperature.getConnectedGraph())
    gradient_optimizer = kn.GradientOptimizer(net)

    logProb = gradient_optimizer.max_likelihood()

    assert logProb < 0.

    temperature = model.temperature.getValue().scalar()

    assert temperature > 20.995 and temperature < 21.005


def test_non_gradient_can_set_builder_properties():
    thermometers = Thermometers()
    model = thermometers.model()
    non_gradient_optimizer = kn.NonGradientOptimizer(model.temperature, 5000, 10., 0.1, 0.2)
    pass


def test_map_non_gradient():
    a = kn.Gaussian(0., 50.)
    b = kn.Gaussian(0., 50.)
    c = a + b
    d = kn.Gaussian(c, 1.)
    d.observe(20.0)

    net = kn.BayesNet(d.getConnectedGraph())
    non_gradient_optimizer = kn.NonGradientOptimizer(net)

    logProb = non_gradient_optimizer.max_a_posteriori()

    assert logProb < 0.

    sum_ab = a.getValue().scalar() + b.getValue().scalar()

    assert sum_ab > 19.9 and sum_ab < 20.1


def test_max_likelihood_non_gradient():
    a = kn.Gaussian(0., 50.)
    b = kn.Gaussian(0., 50.)
    c = a + b
    d = kn.Gaussian(c, 1.)
    d.observe(20.0)

    net = kn.BayesNet(d.getConnectedGraph())
    non_gradient_optimizer = kn.NonGradientOptimizer(net)

    logProb = non_gradient_optimizer.max_likelihood()

    assert logProb < 0.

    sum_ab = a.getValue().scalar() + b.getValue().scalar()

    assert sum_ab > 19.9 and sum_ab < 20.1

from keanu.vertex import Uniform, Gaussian
from keanu import Model
from keanu.algorithm import GradientOptimizer, NonGradientOptimizer

def build_model():
    with Model() as m:
        m.temperature = Uniform(20.,30.)
        m.first_thermometer = Gaussian(m.temperature, 2.5)
        m.second_thermometer = Gaussian(m.temperature, 5.)
    return m

def test_gradient_optimzer_example():
    model = build_model()
    model.first_thermometer.observe(25)
    bayes_net = model.to_bayes_net()
    # %%SNIPPET_START%% PythonGradientOptimizer
    optimizer = GradientOptimizer(bayes_net, max_evaluations=5000,
                                  relative_threshold=1e-8, absolute_threshold=1e-8)
    optimizer.max_a_posteriori()
    calculated_temperature = model.temperature.get_value()
    # %%SNIPPET_END%% PythonGradientOptimizer

def test_non_gradient_optimizer_example():
    model = build_model()
    model.first_thermometer.observe(25)
    bayes_net = model.to_bayes_net()
    # %%SNIPPET_START%% PythonNonGradientOptimizer
    optimizer = NonGradientOptimizer(bayes_net, max_evaluations=5000, bounds_range=100000.,
                                     initial_trust_region_radius=5., stopping_trust_region_radius=2e-8)
    optimizer.max_a_posteriori()
    calculated_temperature = model.temperature.get_value()
    # %%SNIPPET_END%% PythonNonGradientOptimizer
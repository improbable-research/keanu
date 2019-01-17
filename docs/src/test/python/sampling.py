from keanu.algorithm import sample
from keanu import BayesNet, KeanuRandom, Model
from keanu.vertex import Gamma, Exponential, Cauchy, Gaussian
import numpy as np

def inference_example_metropolis():
    # %%SNIPPET_START%% PythonMetropolisExample
    with Model() as m:
        m.a = Gaussian(20.,1.)
        m.b = Gaussian(20.,1.)
        m.c = Gaussian(m.a+m.b,1.)
    m.c.observe(43.)
    m.a.set_value(20.)
    m.b.set_value(20.)
    bayes_net = m.to_bayes_net()
    posterior_samples = sample(net=bayes_net, sample_from=bayes_net.get_latent_vertices(),
                               algo="metropolis", draws=100000)

    average_posterior_a = np.average(posterior_samples.get('a'))
    average_posterior_b = np.average(posterior_samples.get('b'))

    actual = average_posterior_a + average_posterior_b
    # %%SNIPPET_END%% PythonMetropolisExample

def inference_example_hmc_nuts():
    with Model() as m:
        m.a = Gaussian(20., 1.)
        m.b = Gaussian(20., 1.)
        m.c = Gaussian(m.a + m.b, 1.)
    m.c.observe(43.)
    m.a.set_value(20.)
    m.b.set_value(20.)
    bayes_net = m.to_bayes_net()
    # %%SNIPPET_START%% PythonNUTSExample
    posterior_samples = sample(net=bayes_net, sample_from=bayes_net.get_latent_vertices(),
                               algo="NUTS", draws=2000)
    # %%SNIPPET_END%% PythonNUTSExample
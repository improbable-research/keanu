import pandas as pd
import keanu as kn
from examples import CoalMining



def test_coalmining():
    coal_mining = CoalMining()
    model = coal_mining.model()

    model.disasters.observe(coal_mining.training_data())

    net = kn.BayesNet(model.switchpoint.get_connected_graph())
    posterior_dist_samples = kn.MetropolisHastings().get_posterior_samples(net, net.get_latent_vertices(), 50000)
    posterior_dist_samples.drop(10000).down_sample(5)

    switch_year = posterior_dist_samples.get_integer_tensor_samples(model.switchpoint).get_scalar_mode()
    assert switch_year == 1890

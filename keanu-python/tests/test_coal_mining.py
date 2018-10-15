import pandas as pd
import keanu as kn
from examples import CoalMining



def test_coalmining():
    coal_mining = CoalMining()
    model = coal_mining.model()

    model.disasters.observe(coal_mining.training_data())

    net = kn.BayesNet(model.switchpoint.getConnectedGraph())
    posterior_dist_samples = kn.MetropolisHastings().get_posterior_samples(net, net.getLatentVertices(), 50000)
    posterior_dist_samples.drop(10000).downSample(5)

    switch_year = posterior_dist_samples.getIntegerTensorSamples(model.switchpoint.unwrap()).getScalarMode()
    assert switch_year == 1890

import pandas as pd
import keanu as kn
from examples import CoalMining
import statistics

def test_coalmining():
    coal_mining = CoalMining()
    model = coal_mining.model()

    model.disasters.observe(coal_mining.training_data())

    net = kn.BayesNet(model.switchpoint.get_connected_graph())
    samples = kn.sample(net=net, sample_from=net.get_latent_vertices(), draws=50000, drop=10000, down_sample_interval=5)

    switch_year = statistics.mode([sample[0] for sample in samples[model.switchpoint.get_id()]])
    assert switch_year == 1890

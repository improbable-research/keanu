import pandas as pd
import keanu as kn
import examples


def test_coalmining():
    model = examples.coal_mining_model()

    FILE = "data/coal-mining-disaster-data.csv"
    data = pd.read_csv(FILE, names=["year", "count"]).set_index("year")
    model.disasters.observe(data.values)

    net = kn.BayesNet(model.switchpoint.getConnectedGraph())
    posterior_dist_samples = kn.MetropolisHastings().get_posterior_samples(net, net.getLatentVertices(), 50000)
    posterior_dist_samples.drop(10000).downSample(5)

    switch_year = posterior_dist_samples.getIntegerTensorSamples(model.switchpoint.unwrap()).getScalarMode()
    assert switch_year == 1890

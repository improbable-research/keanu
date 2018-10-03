import pandas as pd
import keanu as kn
import numpy as np

def test_coalmining():
    FILE = "data/coal-mining-disaster-data.csv"
    data = pd.read_csv(FILE, names=["year", "count"]).set_index("year")
    start_year, end_year = (data.index.min(), data.index.max())

    with kn.Model() as m:
        m.switchpoint = kn.UniformInt(int(start_year), int(end_year + 1))

        m.early_rate = kn.Exponential(1.0)
        m.late_rate = kn.Exponential(1.0)

        m.years = np.array(data.index)
        m.rates = kn.DoubleIf([1, 1], m.switchpoint > m.years, m.early_rate, m.late_rate)
        m.disasters = kn.Poisson(m.rates)

    m.disasters.observe(data.values)

    net = kn.BayesNet(m.switchpoint.getConnectedGraph())
    posterior_dist_samples = kn.MetropolisHastings().get_posterior_samples(net, net.getLatentVertices(), 50000)
    posterior_dist_samples.drop(10000).downSample(5)

    switch_year = posterior_dist_samples.getIntegerTensorSamples(m.switchpoint.unwrap()).getScalarMode()
    assert switch_year == 1890

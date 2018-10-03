import pandas as pd
import keanu as kn
import numpy as np

def test_coalmining():
    FILE = "data/coal-mining-disaster-data.csv"
    data = pd.read_csv(FILE, names=["year", "count"]).set_index("year")
    start_year, end_year = (data.index.min(), data.index.max())

    with kn.Model() as m:
        m.switchpoint = kn.UniformInt(start_year, end_year + 1)

        m.early_rate = kn.Exponential(1.0)
        m.late_rate = kn.Exponential(1.0)

        m.years = np.array(data.index)
        m.rates = kn.DoubleIf([1, 1], m.switchpoint > m.years, m.early_rate, m.late_rate)
        m.disasters = kn.Poisson(m.rates)

    m.disasters.observe(data.values)

    net = kn.BayesNet(m.switchpoint.getConnectedGraph())
    posterior_dist_samples = kn.MetropolisHastings().get_posterior_samples(net, net.getLatentVertices(), 5000)
    posterior_dist_samples.drop(1000).downSample(5)

    df = kn.samples_to_dataframe(net, posterior_dist_samples, model=m)
    assert len(df) == 5000

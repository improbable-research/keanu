import pandas as pd
import keanu as kn
import numpy as np


def coal_mining_model():
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

    return m

import pandas as pd
import keanu as kn
import numpy as np


class CoalMining():
    __fname = "data/coal-mining-disaster-data.csv"

    def __init__(self):
        self._data = pd.read_csv(CoalMining.__fname, names=["year", "count"]).set_index("year")

    def model(self):
        start_year, end_year = (self._data.index.min(), self._data.index.max())

        with kn.Model() as m:
            m.switchpoint = kn.UniformInt(int(start_year), int(end_year + 1))

            m.early_rate = kn.Exponential(1.0)
            m.late_rate = kn.Exponential(1.0)

            m.years = np.array(self._data.index)
            m.rates = kn.DoubleIf([1, 1], m.switchpoint > m.years, m.early_rate, m.late_rate)
            m.disasters = kn.Poisson(m.rates)

        return m

    def training_data(self):
        return self._data.values

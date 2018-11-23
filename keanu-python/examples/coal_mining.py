import pandas as pd
import numpy as np
from keanu.vertex import UniformInt, Exponential, DoubleIf, Poisson
from keanu import Model


class CoalMining():
    __fname = "data/coal-mining-disaster-data.csv"

    def __init__(self) -> None:
        self._data: pd.DataFrame = pd.read_csv(CoalMining.__fname, names=["year", "count"]).set_index("year") # type: ignore # TODO: stub pandas io

    def model(self) -> Model:
        start_year, end_year = (self._data.index.min(), self._data.index.max())

        with Model() as m:
            m.switchpoint = UniformInt(start_year, end_year + 1)

            m.early_rate = Exponential(1.0)
            m.late_rate = Exponential(1.0)

            m.years = np.array(self._data.index) # type: ignore # because Model#__set_attr__ expects Vertex
            m.rates = DoubleIf(m.switchpoint > m.years, m.early_rate, m.late_rate)
            m.disasters = Poisson(m.rates)

        return m

    def training_data(self) -> np.ndarray:
        return self._data.values

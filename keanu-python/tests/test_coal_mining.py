from typing import List

import numpy as np
from examples import CoalMining
from keanu import BayesNet, KeanuRandom
from keanu.algorithm import sample
from keanu.vartypes import numpy_types, primitive_types


def test_coalmining() -> None:
    KeanuRandom.set_default_random_seed(1)
    coal_mining = CoalMining()
    model = coal_mining.model()

    model.disasters.observe(coal_mining.training_data())

    net = BayesNet(model.switchpoint.get_connected_graph())
    samples = sample(net=net, sample_from=net.get_latent_vertices(), draws=2000, drop=100, down_sample_interval=5)

    vertex_samples: List[numpy_types] = samples["switchpoint"]
    vertex_samples_concatentated: np.ndarray = np.array(vertex_samples)

    switch_year = np.argmax(np.bincount(vertex_samples_concatentated))

    assert switch_year == 1890

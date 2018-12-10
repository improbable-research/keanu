from keanu import Model, BayesNet
from keanu.vertex import Exponential, Gamma, Gaussian


def test_to_bayes_net() -> None:
    with Model() as m:
        m.mu = Exponential(1.)
        m.sigma = Gamma(0.5, 0.1)

        m.gaussian = Gaussian(m.mu, m.sigma)

    net = m.to_bayes_net()

    assert isinstance(net, BayesNet)

    net_vertex_ids = [vertex.get_id() for vertex in net.get_latent_or_observed_vertices()]

    assert len(net_vertex_ids) == 3
    assert m.mu.get_id() in net_vertex_ids
    assert m.sigma.get_id() in net_vertex_ids
    assert m.gaussian.get_id() in net_vertex_ids


def test_to_bayes_net_excludes_non_vertices() -> None:
    with Model() as m:
        m.not_a_vertex = 1
        m.vertex = Gamma(0.5, 0.1)

    net = m.to_bayes_net()

    assert isinstance(net, BayesNet)

    net_vertex_ids = [vertex.get_id() for vertex in net.get_latent_or_observed_vertices()]

    assert len(net_vertex_ids) == 1
    assert m.vertex.get_id() in net_vertex_ids

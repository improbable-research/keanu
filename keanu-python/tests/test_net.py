from keanu.vertex import UniformInt, Gamma, Poisson, Cauchy
from keanu import BayesNet, KeanuRandom
from keanu.network_io import ProtobufLoader, JsonLoader, ProtobufSaver, DotSaver, JsonSaver
import pytest


def test_construct_bayes_net() -> None:
    uniform = UniformInt(0, 1)
    graph = set(uniform.get_connected_graph())
    vertex_ids = [vertex.get_id() for vertex in graph]

    assert len(vertex_ids) == 3
    assert uniform.get_id() in vertex_ids

    net = BayesNet(graph)
    latent_vertex_ids = [vertex.get_id() for vertex in net.get_latent_vertices()]

    assert len(latent_vertex_ids) == 1
    assert uniform.get_id() in latent_vertex_ids


@pytest.mark.parametrize("get_method, latent, observed, continuous, discrete",
                         [("get_latent_or_observed_vertices", True, True, True, True),
                          ("get_latent_vertices", True, False, True, True),
                          ("get_observed_vertices", False, True, True, True),
                          ("get_continuous_latent_vertices", True, False, True, False),
                          ("get_discrete_latent_vertices", True, False, False, True)])
def test_can_get_vertices_from_bayes_net(get_method: str, latent: bool, observed: bool, continuous: bool,
                                         discrete: bool) -> None:
    gamma = Gamma(1., 1.)
    gamma.observe(0.5)

    poisson = Poisson(gamma)
    cauchy = Cauchy(gamma, 1.)

    assert gamma.is_observed()
    assert not poisson.is_observed()
    assert not cauchy.is_observed()

    net = BayesNet([gamma, poisson, cauchy])
    vertex_ids = [vertex.get_id() for vertex in getattr(net, get_method)()]

    if observed and continuous:
        assert gamma.get_id() in vertex_ids
    if latent and discrete:
        assert poisson.get_id() in vertex_ids
    if latent and continuous:
        assert cauchy.get_id() in vertex_ids

    assert len(vertex_ids) == (observed and continuous) + (latent and discrete) + (latent and continuous)


def test_probe_for_non_zero_probability_from_bayes_net() -> None:
    gamma = Gamma(1., 1.)
    poisson = Poisson(gamma)

    net = BayesNet([poisson, gamma])

    assert not gamma.has_value()
    assert not poisson.has_value()

    net.probe_for_non_zero_probability(100, KeanuRandom())

    assert gamma.has_value()
    assert poisson.has_value()


def test_can_save_and_load(tmpdir) -> None:
    PROTO_FILE = str(tmpdir.join("test.proto"))
    JSON_FILE = str(tmpdir.join("test.json"))
    DOT_FILE = str(tmpdir.join("test.dot"))

    gamma = Gamma(1.0, 1.0)
    gamma.set_value(2.5)
    net = BayesNet(gamma.get_connected_graph())
    metadata = {"Team": "GraphOS"}
    protobuf_saver = ProtobufSaver(net)
    protobuf_saver.save(PROTO_FILE, True, metadata)
    json_saver = JsonSaver(net)
    json_saver.save(JSON_FILE, True, metadata)
    dot_saver = DotSaver(net)
    dot_saver.save(DOT_FILE, True, metadata)

    protobuf_loader = ProtobufLoader()
    json_loader = JsonLoader()
    new_net_from_proto = protobuf_loader.load(PROTO_FILE)
    new_net_from_json = json_loader.load(JSON_FILE)
    latents = list(new_net_from_proto.get_latent_vertices())
    assert len(latents) == 1
    new_gamma = latents[0]
    assert new_gamma.get_value() == 2.5
    latents = list(new_net_from_json.get_latent_vertices())
    assert len(latents) == 1
    new_gamma = latents[0]
    assert new_gamma.get_value() == 2.5
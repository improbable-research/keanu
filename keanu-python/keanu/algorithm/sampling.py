from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vertex.base import Vertex
from keanu.net import BayesNet
from typing import Any, Iterable, Dict, List, Tuple, Generator
from keanu.vartypes import sample_types, sample_generator_types

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.NUTS")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.Hamiltonian")

algorithms = {
    'metropolis': k.jvm_view().MetropolisHastings,
    'NUTS': k.jvm_view().NUTS,
    'hamiltonian': k.jvm_view().Hamiltonian
}


def sample(net: BayesNet,
           sample_from: Iterable[Vertex],
           algo: str = 'metropolis',
           draws: int = 500,
           drop: int = 0,
           down_sample_interval: int = 1) -> sample_types:

    vertices_unwrapped = k.to_java_object_list(sample_from)

    network_samples = algorithms[algo].withDefaultConfig().getPosteriorSamples(
        net.unwrap(), vertices_unwrapped, draws).drop(drop).downSample(down_sample_interval)
    vertex_samples = {
        Vertex._get_python_id(vertex_unwrapped): list(
            map(Tensor._to_ndarray,
                network_samples.get(vertex_unwrapped).asList())) for vertex_unwrapped in vertices_unwrapped
    }

    return vertex_samples


def generate_samples(net: BayesNet,
                     sample_from: Iterable[Vertex],
                     algo: str = 'metropolis',
                     drop: int = 0,
                     down_sample_interval: int = 1) -> sample_generator_types:
    vertices_unwrapped = k.to_java_object_list(sample_from)

    sample_iterator = algorithms[algo].withDefaultConfig().generatePosteriorSamples(net.unwrap(), vertices_unwrapped)
    sample_iterator = sample_iterator.dropCount(drop).downSampleInterval(down_sample_interval)
    sample_iterator = sample_iterator.stream().iterator()

    return _samples_generator(sample_iterator, vertices_unwrapped)


def _samples_generator(sample_iterator: Any, vertices_unwrapped: Any) -> sample_generator_types:
    while (True):
        network_sample = sample_iterator.next()
        sample = {
            Vertex._get_python_id(vertex_unwrapped): Tensor._to_ndarray(network_sample.get(vertex_unwrapped))
            for vertex_unwrapped in vertices_unwrapped
        }
        yield sample

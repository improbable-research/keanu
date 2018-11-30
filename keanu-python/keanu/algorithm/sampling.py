from py4j.java_gateway import java_import, JavaObject
from py4j.java_collections import JavaList

from keanu.algorithm._proposal_distribution import ProposalDistribution
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vertex.base import Vertex
from keanu.net import BayesNet
from typing import Any, Iterable, Dict, List, Tuple, Generator
from keanu.vartypes import sample_types, sample_generator_types, numpy_types

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
           proposal_distribution: str = None,
           proposal_distribution_sigma: numpy_types = None,
           proposal_listeners=[],
           draws: int = 500,
           drop: int = 0,
           down_sample_interval: int = 1) -> sample_types:

    sampling_algorithm: JavaObject = build_sampling_algorithm(
        algo,
        proposal_distribution,
        proposal_distribution_sigma,
        proposal_listeners
    )

    vertices_unwrapped: JavaList = k.to_java_object_list(sample_from)

    network_samples: JavaObject = sampling_algorithm.getPosteriorSamples(net.unwrap(), vertices_unwrapped, draws).drop(drop).downSample(down_sample_interval)

    vertex_samples = {
        Vertex._get_python_id(vertex_unwrapped): list(
            map(Tensor._to_ndarray,
                network_samples.get(vertex_unwrapped).asList())) for vertex_unwrapped in vertices_unwrapped
    }

    return vertex_samples


def generate_samples(net: BayesNet,
                     sample_from: Iterable[Vertex],
                     algo: str = 'metropolis',
                     proposal_distribution: str = None,
                     proposal_distribution_sigma: numpy_types = None,
                     proposal_listeners=[],
                     drop: int = 0,
                     down_sample_interval: int = 1) -> sample_generator_types:

    sampling_algorithm: JavaObject = build_sampling_algorithm(
        algo,
        proposal_distribution,
        proposal_distribution_sigma,
        proposal_listeners
    )

    vertices_unwrapped: JavaList = k.to_java_object_list(sample_from)

    samples: JavaObject = sampling_algorithm.generatePosteriorSamples(net.unwrap(), vertices_unwrapped)
    samples = samples.dropCount(drop).downSampleInterval(down_sample_interval)
    sample_iterator: JavaObject = samples.stream().iterator()

    return _samples_generator(sample_iterator, vertices_unwrapped)


def build_sampling_algorithm(algo, proposal_distribution, proposal_distribution_sigma, proposal_listeners):
    if (algo != "metropolis" and proposal_distribution is not None):
        raise TypeError("Only Metropolis Hastings supports the proposal_distribution parameter")

    builder: JavaObject = algorithms[algo].builder()
    if proposal_distribution is not None:
        proposal_distribution_object = ProposalDistribution(
            type_=proposal_distribution, sigma=proposal_distribution_sigma, listeners=proposal_listeners)
        builder = builder.proposalDistribution(proposal_distribution_object.unwrap())
    sampling_algorithm: JavaObject = builder.build()
    return sampling_algorithm


def _samples_generator(sample_iterator: JavaObject, vertices_unwrapped: JavaList) -> sample_generator_types:
    while (True):
        network_sample = sample_iterator.next()
        sample = {
            Vertex._get_python_id(vertex_unwrapped): Tensor._to_ndarray(network_sample.get(vertex_unwrapped))
            for vertex_unwrapped in vertices_unwrapped
        }
        yield sample

from py4j.java_gateway import java_import, JavaObject
from py4j.java_collections import JavaList

from keanu.algorithm._proposal_distribution import ProposalDistribution
from keanu.algorithm.proposal_listeners import proposal_listener_types
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vertex.base import Vertex
from keanu.net import BayesNet
from typing import Any, Iterable, Dict, List, Tuple, Generator, Optional
from keanu.vartypes import sample_types, sample_generator_types, numpy_types
from keanu.plots import traceplot

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
           down_sample_interval: int = 1,
           plot: bool = False,
           ax: Any = None) -> sample_types:

    sampling_algorithm: JavaObject = build_sampling_algorithm(algo, proposal_distribution, proposal_distribution_sigma,
                                                              proposal_listeners)

    vertices_unwrapped: JavaList = k.to_java_object_list(sample_from)

    network_samples: JavaObject = sampling_algorithm.getPosteriorSamples(
        net.unwrap(), vertices_unwrapped, draws).drop(drop).downSample(down_sample_interval)

    vertex_samples = {
        Vertex._get_python_label(vertex_unwrapped): list(
            map(Tensor._to_ndarray,
                network_samples.get(vertex_unwrapped).asList())) for vertex_unwrapped in vertices_unwrapped
    }

    if plot:
        traceplot(vertex_samples, ax=ax)

    return vertex_samples


def generate_samples(net: BayesNet,
                     sample_from: Iterable[Vertex],
                     algo: str = 'metropolis',
                     proposal_distribution: str = None,
                     proposal_distribution_sigma: numpy_types = None,
                     proposal_listeners: List[proposal_listener_types] = [],
                     drop: int = 0,
                     down_sample_interval: int = 1,
                     live_plot: bool = False,
                     refresh_every: int = 100,
                     ax: Any = None) -> sample_generator_types:

    sampling_algorithm: JavaObject = build_sampling_algorithm(algo, proposal_distribution, proposal_distribution_sigma,
                                                              proposal_listeners)

    vertices_unwrapped: JavaList = k.to_java_object_list(sample_from)

    samples: JavaObject = sampling_algorithm.generatePosteriorSamples(net.unwrap(), vertices_unwrapped)
    samples = samples.dropCount(drop).downSampleInterval(down_sample_interval)
    sample_iterator: JavaObject = samples.stream().iterator()

    return _samples_generator(
        sample_iterator, vertices_unwrapped, live_plot=live_plot, refresh_every=refresh_every, ax=ax)


def build_sampling_algorithm(algo, proposal_distribution: Optional[str],
                             proposal_distribution_sigma: Optional[numpy_types],
                             proposal_listeners: List[proposal_listener_types]):
    if algo != "metropolis":
        if proposal_distribution is not None:
            raise TypeError("Only Metropolis Hastings supports the proposal_distribution parameter")
        if len(proposal_listeners) > 0:
            raise TypeError("Only Metropolis Hastings supports the proposal_listeners parameter")

    if (proposal_distribution is None and len(proposal_listeners) > 0):
        raise TypeError("If you pass in proposal_listeners you must also specify proposal_distribution")

    builder: JavaObject = algorithms[algo].builder()

    if proposal_distribution is not None:
        proposal_distribution_object = ProposalDistribution(
            type_=proposal_distribution, sigma=proposal_distribution_sigma, listeners=proposal_listeners)
        builder = builder.proposalDistribution(proposal_distribution_object.unwrap())
    sampling_algorithm: JavaObject = builder.build()
    return sampling_algorithm


def _samples_generator(sample_iterator: JavaObject, vertices_unwrapped: JavaList, live_plot: bool, refresh_every: int,
                       ax: Any) -> sample_generator_types:
    traces = []
    x0 = 0
    while (True):
        network_sample = sample_iterator.next()
        sample = {
            Vertex._get_python_label(vertex_unwrapped): Tensor._to_ndarray(network_sample.get(vertex_unwrapped))
            for vertex_unwrapped in vertices_unwrapped
        }

        if live_plot:
            traces.append(sample)
            if len(traces) % refresh_every == 0:
                joined_trace = {k: [t[k] for t in traces] for k in sample.keys()}
                if ax is None:
                    ax = traceplot(joined_trace, x0=x0)
                else:
                    traceplot(joined_trace, ax=ax, x0=x0)
                x0 += refresh_every
                traces = []

        yield sample

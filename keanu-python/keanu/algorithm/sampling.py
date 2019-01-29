from py4j.java_gateway import java_import, JavaObject
from py4j.java_collections import JavaList

from keanu.algorithm._proposal_distribution import ProposalDistribution
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vertex.base import Vertex
from keanu.net import BayesNet
from typing import Any, Iterable, Dict, List, Tuple, Generator, Optional
from keanu.vartypes import sample_types, sample_generator_types, numpy_types
from keanu.plots import traceplot
from collections import defaultdict
from numpy import ndenumerate, ndarray
import itertools
from itertools import tee

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.nuts.NUTS")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.Hamiltonian")


class PosteriorSamplingAlgorithm:

    def __init__(self, sampler: JavaObject):
        self._sampler = sampler

    def get_sampler(self) -> JavaObject:
        return self._sampler


class MetropolisHastingsSampler(PosteriorSamplingAlgorithm):

    def __init__(self,
                 proposal_distribution: str = None,
                 proposal_distribution_sigma: numpy_types = None,
                 proposal_listeners=[],
                 use_cache_on_rejection: bool = None):

        if (proposal_distribution is None and len(proposal_listeners) > 0):
            raise TypeError("If you pass in proposal_listeners you must also specify proposal_distribution")

        builder: JavaObject = k.jvm_view().MetropolisHastings.builder()

        if proposal_distribution is not None:
            proposal_distribution_object = ProposalDistribution(
                type_=proposal_distribution, sigma=proposal_distribution_sigma, listeners=proposal_listeners)
            builder = builder.proposalDistribution(proposal_distribution_object.unwrap())

        if use_cache_on_rejection is not None:
            builder.useCacheOnRejection(use_cache_on_rejection)

        super().__init__(builder.build())


class HamiltonianSampler(PosteriorSamplingAlgorithm):

    def __init__(self) -> None:
        super().__init__(k.jvm_view().Hamiltonian.withDefaultConfig())


class NUTSSampler(PosteriorSamplingAlgorithm):

    def __init__(self,
                 adapt_count: int = None,
                 target_acceptance_prob: float = None,
                 adapt_enabled: bool = None,
                 initial_step_size: float = None,
                 max_tree_height: int = None):

        builder: JavaObject = k.jvm_view().NUTS.builder()

        if adapt_count is not None:
            builder.adaptCount(adapt_count)

        if target_acceptance_prob is not None:
            builder.targetAcceptanceProb(target_acceptance_prob)

        if adapt_enabled is not None:
            builder.adaptEnabled(adapt_enabled)

        if initial_step_size is not None:
            builder.initialStepSize(initial_step_size)

        if max_tree_height is not None:
            builder.maxTreeHeight(max_tree_height)

        super().__init__(builder.build())


def sample(net: BayesNet,
           sample_from: Iterable[Vertex],
           sampling_algorithm: PosteriorSamplingAlgorithm = None,
           draws: int = 500,
           drop: int = 0,
           down_sample_interval: int = 1,
           plot: bool = False,
           ax: Any = None) -> sample_types:

    if sampling_algorithm is None:
        sampling_algorithm = MetropolisHastingsSampler()

    vertices_unwrapped: JavaList = k.to_java_object_list(sample_from)

    network_samples: JavaObject = sampling_algorithm.get_sampler().getPosteriorSamples(
        net.unwrap(), vertices_unwrapped, draws).drop(drop).downSample(down_sample_interval)

    vertex_samples = {
        Vertex._get_python_label(vertex_unwrapped): list(
            map(Tensor._to_ndarray,
                network_samples.get(vertex_unwrapped).asList(),
                itertools.repeat(True, len(network_samples.get(vertex_unwrapped).asList()))))
        for vertex_unwrapped in vertices_unwrapped
    }

    if plot:
        traceplot(vertex_samples, ax=ax)

    if _all_samples_are_scalar(vertex_samples):
        return vertex_samples
    else:
        return _create_multi_indexed_samples(vertex_samples, False)


def generate_samples(net: BayesNet,
                     sample_from: Iterable[Vertex],
                     sampling_algorithm: PosteriorSamplingAlgorithm = None,
                     drop: int = 0,
                     down_sample_interval: int = 1,
                     live_plot: bool = False,
                     refresh_every: int = 100,
                     ax: Any = None) -> sample_generator_types:

    if sampling_algorithm is None:
        sampling_algorithm = MetropolisHastingsSampler()

    sample_from, sample_from_copy = tee(sample_from)
    vertices_unwrapped: JavaList = k.to_java_object_list(sample_from)

    samples: JavaObject = sampling_algorithm.get_sampler().generatePosteriorSamples(net.unwrap(), vertices_unwrapped)
    samples = samples.dropCount(drop).downSampleInterval(down_sample_interval)
    sample_iterator: JavaObject = samples.stream().iterator()

    all_are_scalar = _all_vertices_are_scalar_generator(sample_from_copy)
    return _samples_generator(
        sample_iterator, vertices_unwrapped, live_plot=live_plot, refresh_every=refresh_every, ax=ax, all_scalar=all_are_scalar)


def _all_samples_are_scalar(samples: dict) -> bool:
    for vertex_label in samples:
        if type(samples[vertex_label][0]) is ndarray:
            return False
    return True


def _all_vertices_are_scalar_generator(sample_from: Iterable[Vertex]) -> bool:
    for vertex in sample_from:
        if vertex.get_value().shape != ():
            return False
    return True


def _samples_generator(sample_iterator: JavaObject, vertices_unwrapped: JavaList, live_plot: bool, refresh_every: int,
                       ax: Any, all_scalar: bool) -> sample_generator_types:
    traces = []
    x0 = 0
    while (True):
        network_sample = sample_iterator.next()
        sample = {
            Vertex._get_python_label(vertex_unwrapped): Tensor._to_ndarray(network_sample.get(vertex_unwrapped), primitive=True)
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

        if all_scalar:
            yield sample
        else:
            yield _create_multi_indexed_samples(sample, True)


def _create_multi_indexed_samples(samples: dict, generated: bool) -> dict:
    vertex_samples_multi: dict = {}
    column_header_for_scalar = '(0)'
    for vertex_label in samples:
        vertex_samples_multi[vertex_label] = defaultdict(list)
        if not generated:
            for sample_value in samples[vertex_label]:
                _add_sample_to_dict(sample_value, vertex_samples_multi, vertex_label, column_header_for_scalar)
        else:
            _add_sample_to_dict(samples[vertex_label], vertex_samples_multi, vertex_label, column_header_for_scalar)

        tuple_heirarchy = {(vertex_label, tensor_index): values
                       for vertex_label, tensor_index in vertex_samples_multi.items()
                       for tensor_index, values in tensor_index.items()}

    return tuple_heirarchy

def _add_sample_to_dict(sample_value: Any, vertex_samples_multi: dict, vertex_label: str, column_header_for_scalar: str):
    if type(sample_value) is not ndarray:
        vertex_samples_multi[vertex_label][column_header_for_scalar].append(sample_value)
    elif sample_value.shape == ():
        vertex_samples_multi[vertex_label][column_header_for_scalar].append(sample_value.item())
    else:
        for index, value in ndenumerate(sample_value):
            vertex_samples_multi[vertex_label][str(index)].append(value.item())
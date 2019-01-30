from py4j.java_gateway import java_import, JavaObject
from py4j.java_collections import JavaList

from keanu.algorithm._proposal_distribution import ProposalDistribution
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vertex.base import Vertex
from keanu.net import BayesNet, ProbabilisticModel, ProbabilisticModelWithGradient
from typing import Any, Iterable, Dict, Tuple, Union
from keanu.vartypes import sample_types, sample_generator_types, numpy_types, primitive_types
from keanu.plots import traceplot
from collections import defaultdict
from numpy import ndenumerate, ndarray
import itertools
from itertools import tee

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.nuts.NUTS")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.RollBackToCachedValuesOnRejection")


class PosteriorSamplingAlgorithm:

    def __init__(self, sampler: JavaObject):
        self._sampler = sampler

    def get_sampler(self) -> JavaObject:
        return self._sampler


class MetropolisHastingsSampler(PosteriorSamplingAlgorithm):

    def __init__(self,
                 proposal_distribution: str,
                 latents: Iterable[Vertex],
                 proposal_listeners=[],
                 proposal_distribution_sigma: numpy_types = None):

        if (proposal_distribution is None and len(proposal_listeners) > 0):
            raise TypeError("If you pass in proposal_listeners you must also specify proposal_distribution")

        builder: JavaObject = k.jvm_view().MetropolisHastings.builder()

        latents, latents_copy = tee(latents)

        proposal_distribution_object = ProposalDistribution(
            type_=proposal_distribution,
            sigma=proposal_distribution_sigma,
            latents=latents_copy,
            listeners=proposal_listeners)

        rejection_strategy = k.jvm_view().RollBackToCachedValuesOnRejection(k.to_java_object_list(latents))

        builder = builder.proposalDistribution(proposal_distribution_object.unwrap())
        builder = builder.rejectionStrategy(rejection_strategy)

        super().__init__(builder.build())


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

    sample_from = list(sample_from)

    if sampling_algorithm is None:
        sampling_algorithm = MetropolisHastingsSampler(proposal_distribution="prior", latents=sample_from)

    sample_from = list(sample_from)
    vertices_unwrapped: JavaList = k.to_java_object_list(sample_from)

    probabilistic_model = ProbabilisticModel(net) if isinstance(
        sampling_algorithm, MetropolisHastingsSampler) else ProbabilisticModelWithGradient(net)

    network_samples: JavaObject = sampling_algorithm.get_sampler().getPosteriorSamples(
        probabilistic_model.unwrap(), vertices_unwrapped, draws).drop(drop).downSample(down_sample_interval)

    vertex_samples: sample_types = {
        Vertex._get_python_label(vertex_unwrapped): list(
            map(lambda samples : Tensor._to_ndarray(samples, True),
                network_samples.get(vertex_unwrapped).asList()))
        for vertex_unwrapped in vertices_unwrapped
    }

    if plot:
        traceplot(vertex_samples, ax=ax)

    if _all_vertices_are_scalar(sample_from):
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

    sample_from = list(sample_from)

    if sampling_algorithm is None:
        sampling_algorithm = MetropolisHastingsSampler(proposal_distribution="prior", latents=sample_from)

    vertices_unwrapped: JavaList = k.to_java_object_list(sample_from)

    probabilistic_model = ProbabilisticModel(net) if isinstance(
        sampling_algorithm, MetropolisHastingsSampler) else ProbabilisticModelWithGradient(net)
    samples: JavaObject = sampling_algorithm.get_sampler().generatePosteriorSamples(probabilistic_model.unwrap(),
                                                                                    vertices_unwrapped)
    samples = samples.dropCount(drop).downSampleInterval(down_sample_interval)
    sample_iterator: JavaObject = samples.stream().iterator()

    all_are_scalar = _all_vertices_are_scalar(sample_from)

    return _samples_generator(
        sample_iterator,
        vertices_unwrapped,
        live_plot=live_plot,
        refresh_every=refresh_every,
        ax=ax,
        all_scalar=all_are_scalar)


def _all_vertices_are_scalar(sample_from: Iterable[Vertex]) -> bool:
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
        sample: Dict[Union[str, Tuple[str, str]], primitive_types] = {
            Vertex._get_python_label(vertex_unwrapped): Tensor._to_ndarray(
                network_sample.get(vertex_unwrapped), primitive=True) for vertex_unwrapped in vertices_unwrapped
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


def _add_sample_to_dict(sample_value: Any, vertex_samples_multi: dict, vertex_label: str,
                        column_header_for_scalar: str):
    if type(sample_value) is not ndarray:
        vertex_samples_multi[vertex_label][column_header_for_scalar].append(sample_value)
    elif sample_value.shape == ():
        vertex_samples_multi[vertex_label][column_header_for_scalar].append(sample_value.item())
    else:
        for index, value in ndenumerate(sample_value):
            vertex_samples_multi[vertex_label][str(index)].append(value.item())

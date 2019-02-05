from collections import defaultdict
from typing import Any, Iterable, Dict, List, Tuple

from numpy import ndenumerate, ndarray
from py4j.java_collections import JavaList
from py4j.java_gateway import java_import, JavaObject

from keanu.algorithm._proposal_distribution import ProposalDistribution
from keanu.context import KeanuContext
from keanu.net import BayesNet, ProbabilisticModel, ProbabilisticModelWithGradient
from keanu.plots import traceplot
from keanu.tensor import Tensor
from keanu.vartypes import sample_types, sample_generator_types, numpy_types, sample_generator_dict_type
from keanu.vertex.base import Vertex

COLUMN_HEADER_FOR_SCALAR = (0,)

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

        latents = list(latents)

        proposal_distribution_object = ProposalDistribution(
            type_=proposal_distribution,
            sigma=proposal_distribution_sigma,
            latents=latents,
            listeners=proposal_listeners)

        rejection_strategy = k.jvm_view().RollBackToCachedValuesOnRejection()

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

    vertices_unwrapped: JavaList = k.to_java_object_list(sample_from)

    probabilistic_model = ProbabilisticModel(net) if isinstance(
        sampling_algorithm, MetropolisHastingsSampler) else ProbabilisticModelWithGradient(net)

    network_samples: JavaObject = sampling_algorithm.get_sampler().getPosteriorSamples(
        probabilistic_model.unwrap(), vertices_unwrapped, draws).drop(drop).downSample(down_sample_interval)

    id_to_label = __check_if_vertices_are_labelled(sample_from)
    if __all_scalar(sample_from):
        vertex_samples = __create_single_indexed_samples(network_samples, vertices_unwrapped, id_to_label)
    else:
        vertex_samples = __create_multi_indexed_samples(vertices_unwrapped, network_samples, id_to_label)

    if plot:
        traceplot(vertex_samples, ax=ax)

    return vertex_samples


def generate_samples(net: BayesNet,
                     sample_from: Iterable[Vertex],
                     sampling_algorithm: PosteriorSamplingAlgorithm = None,
                     drop: int = 0,
                     down_sample_interval: int = 1,
                     live_plot: bool = False,
                     refresh_every: int = 100,
                     ax: Any = None) -> sample_generator_types:

    sample_from = list(sample_from)
    id_to_label = __check_if_vertices_are_labelled(sample_from)

    if sampling_algorithm is None:
        sampling_algorithm = MetropolisHastingsSampler(proposal_distribution="prior", latents=sample_from)

    vertices_unwrapped: JavaList = k.to_java_object_list(sample_from)

    probabilistic_model = ProbabilisticModel(net) if isinstance(
        sampling_algorithm, MetropolisHastingsSampler) else ProbabilisticModelWithGradient(net)
    samples: JavaObject = sampling_algorithm.get_sampler().generatePosteriorSamples(probabilistic_model.unwrap(),
                                                                                    vertices_unwrapped)
    samples = samples.dropCount(drop).downSampleInterval(down_sample_interval)
    sample_iterator: JavaObject = samples.stream().iterator()

    all_are_scalar = __all_scalar(sample_from)

    return _samples_generator(
        sample_iterator,
        vertices_unwrapped,
        live_plot=live_plot,
        refresh_every=refresh_every,
        ax=ax,
        all_scalar=all_are_scalar,
        id_to_label=id_to_label)


def __all_scalar(sample_from: List[Vertex]):
    return not any((vertex.get_value().shape != () for vertex in sample_from))


def _samples_generator(sample_iterator: JavaObject, vertices_unwrapped: JavaList, live_plot: bool, refresh_every: int,
                       ax: Any, all_scalar: bool, id_to_label: Dict[Tuple[int, ...], str]) -> sample_generator_types:
    traces = []
    x0 = 0
    while (True):
        network_sample = sample_iterator.next()

        if all_scalar:
            sample: sample_generator_dict_type = {
                id_to_label[Vertex._get_python_id(vertex_unwrapped)]: Tensor._to_scalar_or_ndarray(
                    network_sample.get(vertex_unwrapped), return_as_primitive=True)
                for vertex_unwrapped in vertices_unwrapped
            }
        else:
            sample = __create_multi_indexed_samples_generated(vertices_unwrapped, network_sample, id_to_label)

        if live_plot:
            traces.append(sample)
            if len(traces) % refresh_every == 0:
                joined_trace: sample_types = {k: [t[k] for t in traces] for k in sample.keys()}
                if ax is None:
                    ax = traceplot(joined_trace, x0=x0)
                else:
                    traceplot(joined_trace, ax=ax, x0=x0)
                x0 += refresh_every
                traces = []

        yield sample


def __check_if_vertices_are_labelled(vertices: List[Vertex]) -> Dict[Tuple[int, ...], str]:
    id_to_label = {}
    for vertex in vertices:
        label = vertex.get_label()
        if label is None:
            raise ValueError("Vertices in sample_from must be labelled.")
        else:
            id_to_label[vertex.get_id()] = label
    return id_to_label


def __create_single_indexed_samples(network_samples: JavaObject, vertices_unwrapped: JavaList,
                                    id_to_label: Dict[Tuple[int, ...], str]) -> sample_types:
    vertex_samples: sample_types = {}
    for vertex_unwrapped in vertices_unwrapped:
        vertex_label = id_to_label[Vertex._get_python_id(vertex_unwrapped)]
        samples_for_vertex = network_samples.get(vertex_unwrapped).asList()
        is_primitive = [True] * len(samples_for_vertex)
        samples_as_ndarray = map(Tensor._to_scalar_or_ndarray, samples_for_vertex, is_primitive)
        vertex_samples[vertex_label] = list(samples_as_ndarray)
    return vertex_samples


def __create_multi_indexed_samples(vertices_unwrapped: JavaList, network_samples: JavaObject,
                                   id_to_label: Dict[Tuple[int, ...], str]) -> sample_types:
    vertex_samples_multi: Dict = {}
    for vertex in vertices_unwrapped:
        vertex_label = id_to_label[Vertex._get_python_id(vertex)]
        vertex_samples_multi[vertex_label] = defaultdict(list)
        samples_for_vertex = network_samples.get(vertex).asList()
        is_primitive = [True] * len(samples_for_vertex)
        samples_as_ndarray = map(Tensor._to_scalar_or_ndarray, samples_for_vertex, is_primitive)
        samples = list(samples_as_ndarray)
        for sample in samples:
            __add_sample_to_dict(sample, vertex_samples_multi[vertex_label])

    tuple_hierarchy: Dict = {(vertex_label, shape_index): values
                             for vertex_label, tensor_index in vertex_samples_multi.items()
                             for shape_index, values in tensor_index.items()}

    return tuple_hierarchy


def __create_multi_indexed_samples_generated(vertices_unwrapped: JavaList, network_samples: JavaObject,
                                             id_to_label: Dict[Tuple[int, ...], str]) -> sample_generator_dict_type:
    vertex_samples_multi: Dict = {}
    for vertex in vertices_unwrapped:
        vertex_label = id_to_label[Vertex._get_python_id(vertex)]
        vertex_samples_multi[vertex_label] = defaultdict(list)
        sample = Tensor._to_scalar_or_ndarray(network_samples.get(vertex), return_as_primitive=True)
        __add_sample_to_dict(sample, vertex_samples_multi[vertex_label])

    tuple_hierarchy: Dict = {(vertex_label, tensor_index): values
                             for vertex_label, tensor_index in vertex_samples_multi.items()
                             for tensor_index, values in tensor_index.items()}

    return tuple_hierarchy


def __add_sample_to_dict(sample_value: Any, vertex_sample: Dict):
    if type(sample_value) is not ndarray:
        vertex_sample[COLUMN_HEADER_FOR_SCALAR].append(sample_value)
    elif sample_value.shape == ():
        vertex_sample[COLUMN_HEADER_FOR_SCALAR].append(sample_value.item())
    else:
        for index, value in ndenumerate(sample_value):
            vertex_sample[index].append(value.item())

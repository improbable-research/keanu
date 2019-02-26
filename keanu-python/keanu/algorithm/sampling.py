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
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.sampling.Forward")


class PosteriorSamplingAlgorithm:

    def __init__(self, sampler: JavaObject):
        self._sampler = sampler

    def get_sampler(self) -> JavaObject:
        return self._sampler


class ForwardSampler(PosteriorSamplingAlgorithm):

    def __init__(self) -> None:
        super().__init__(k.jvm_view().Forward.builder().build())


class MetropolisHastingsSampler(PosteriorSamplingAlgorithm):
    """
    :param proposal_distribution: The proposal distribution for Metropolis Hastings. Options are 'gaussian' and 'prior'.
    :param latents: All latent vertices.
    :param proposal_listeners: Listeners for proposal creation and rejection. Options are :class:`keanu.algorithm.AcceptanceRateTracker`.
    :param proposal_distribution_sigma: Parameter sigma for 'gaussian' proposal distribution.

    :raises TypeError: If you pass `proposal_listener` without specifying `proposal_distribution`.
    :raises TypeError: If you choose 'gaussian' as `proposal_distribution` but did not specify `proposal_distribution_sigma`.
    :raises TypeError: If you choose 'prior' as `proposal_distribution` but did not pass latent vertices.
    """

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
    """
    :param adapt_count: The number of samples for which the step size will be tuned. For the remaining samples in which it is not tuned, the step size will be frozen to its last calculated value. Defaults to 1000.
    :param adapt_step_size_enabled: enable the step size adaption. Defaults to true.
    :param adapt_potential_enabled: enable the potential adaption. Defaults to true.
    :param target_acceptance_prob: The target acceptance probability. Defaults to 0.8.
    :param initial_step_size: Sets the initial step size. If none is given then a heuristic will be used to determine a good step size.
    :param max_energy_change: the maximum energy change before a step is considered divergent.
    :param max_tree_height: The maximum tree size for the sampler. This controls how long a sample walk can be before it terminates. This will set at a maximum approximately 2^treeSize number of logProb evaluations for a sample. Defaults to 10.
    """

    def __init__(self,
                 adapt_count: int = None,
                 adapt_step_size_enabled: bool = None,
                 adapt_potential_enabled: bool = None,
                 target_acceptance_prob: float = None,
                 initial_step_size: float = None,
                 potential_adapt_window_size: int = None,
                 max_energy_change: float = None,
                 max_tree_height: int = None):

        builder: JavaObject = k.jvm_view().NUTS.builder()

        if adapt_count is not None:
            builder.adaptCount(adapt_count)

        if target_acceptance_prob is not None:
            builder.targetAcceptanceProb(target_acceptance_prob)

        if adapt_step_size_enabled is not None:
            builder.adaptStepSizeEnabled(adapt_step_size_enabled)

        if adapt_potential_enabled is not None:
            builder.adaptPotentialEnabled(adapt_potential_enabled)

        if potential_adapt_window_size is not None:
            builder.potentialAdaptWindowSize(potential_adapt_window_size)

        if initial_step_size is not None:
            builder.initialStepSize(initial_step_size)

        if max_energy_change is not None:
            builder.maxEnergyChange(max_energy_change)

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
    """
    :param net: Bayesian Network containing latent variables.
    :param sample_from: Vertices to include in the returned samples.
    :param sampling_algorithm: The posterior sampling algorithm to use.
        Options are :class:`keanu.algorithm.MetropolisHastingsSampler`, :class:`keanu.algorithm.NUTSSampler` and :class:`keanu.algorithm.ForwardSampler`
        If not set, :class:`keanu.algorithm.MetropolisHastingsSampler` is chosen with 'prior' as its proposal distribution.
    :param draws: The number of samples to take.
    :param drop: The number of samples to drop before collecting anything.
        If this is zero then no samples will be dropped before collecting.
    :param down_sample_interval: Collect 1 sample for every `down_sample_interval`.
        If this is 1 then there will be no down-sampling.
        If this is 2 then every other sample will be taken.
        If this is 3 then 2 samples will be dropped before one is taken.
        And so on.
    :param plot: Flag for plotting the trace after sampling.
        Call `matplotlib.pyplot.show <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.show.html>`_ to display the plot.
    :param Axes ax: `matplotlib.axes.Axes <https://matplotlib.org/api/axes_api.html>`_.
        If not set, a new one is created.

    :raises ValueError: If `sample_from` contains vertices without labels.

    :return: Dictionary of samples at an index (tuple) for each vertex label (str). If all the vertices in `sample_from` are scalar, the dictionary is only keyed by label.
    """

    sample_from = list(sample_from)

    if sampling_algorithm is None:
        sampling_algorithm = MetropolisHastingsSampler(proposal_distribution="prior", latents=sample_from)

    vertices_unwrapped: JavaList = k.to_java_object_list(sample_from)

    probabilistic_model = ProbabilisticModel(net) if (
        isinstance(sampling_algorithm, MetropolisHastingsSampler) or
        isinstance(sampling_algorithm, ForwardSampler)) else ProbabilisticModelWithGradient(net)

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
    """
    :param net: Bayesian Network containing latent variables.
    :param sample_from: Vertices to include in the returned samples.
    :param sampling_algorithm: The posterior sampling algorithm to use.
        Options are :class:`keanu.algorithm.MetropolisHastingsSampler` and :class:`keanu.algorithm.NUTSSampler`.
        If not set, :class:`keanu.algorithm.MetropolisHastingsSampler` is chosen with 'prior' as its proposal distribution.
    :param drop: The number of samples to drop before collecting anything.
        If this is zero then no samples will be dropped before collecting.
    :param down_sample_interval: Collect 1 sample for every `down_sample_interval`.
        If this is 1 then there will be no down-sampling.
        If this is 2 then every other sample will be taken.
        If this is 3 then 2 samples will be dropped before one is taken.
    :param live_plot: Flag for plotting the trace while sampling.
        Call `matplotlib.pyplot.show <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.show.html>`_ to display the plot.
    :param refresh_every: Period of plot updates (in sample number).
    :param Axes ax: `matplotlib.axes.Axes <https://matplotlib.org/api/axes_api.html>`_.
        If not set, a new one is created.

    :raises ValueError: If `sample_from` contains vertices without labels.

    :return: Dictionary of samples at an index (tuple) for each vertex label (str). If all the vertices in `sample_from` are scalar, the dictionary is only keyed by label.
    """
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
                id_to_label[Vertex._get_python_id(vertex_unwrapped)]: Tensor._to_ndarray(
                    network_sample.get(vertex_unwrapped)).item() for vertex_unwrapped in vertices_unwrapped
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
        samples_for_vertex = __get_vertex_samples(network_samples, vertex_unwrapped)
        vertex_samples[vertex_label] = samples_for_vertex.tolist()
    return vertex_samples


def __create_multi_indexed_samples(vertices_unwrapped: JavaList, network_samples: JavaObject,
                                   id_to_label: Dict[Tuple[int, ...], str]) -> sample_types:
    vertex_samples_multi: Dict = {}
    for vertex in vertices_unwrapped:
        vertex_label = id_to_label[Vertex._get_python_id(vertex)]
        vertex_samples_multi[vertex_label] = defaultdict(list)
        samples_for_vertex = __get_vertex_samples(network_samples, vertex)
        for sample in samples_for_vertex:
            __add_sample_to_dict(sample, vertex_samples_multi[vertex_label])

    tuple_hierarchy: Dict = {(vertex_label, shape_index): values
                             for vertex_label, samples in vertex_samples_multi.items()
                             for shape_index, values in samples.items()}

    return tuple_hierarchy


def __create_multi_indexed_samples_generated(vertices_unwrapped: JavaList, network_samples: JavaObject,
                                             id_to_label: Dict[Tuple[int, ...], str]) -> sample_generator_dict_type:
    vertex_samples_multi: Dict = {}
    for vertex in vertices_unwrapped:
        vertex_label = id_to_label[Vertex._get_python_id(vertex)]
        vertex_samples_multi[vertex_label] = defaultdict(list)
        sample = Tensor._to_ndarray(network_samples.get(vertex))
        __add_sample_to_dict(sample, vertex_samples_multi[vertex_label])

    tuple_hierarchy: Dict = {(vertex_label, tensor_index): values
                             for vertex_label, tensor_index in vertex_samples_multi.items()
                             for tensor_index, values in tensor_index.items()}

    return tuple_hierarchy


def __add_sample_to_dict(sample_value: Any, vertex_sample: Dict):
    if sample_value.shape == ():
        vertex_sample[COLUMN_HEADER_FOR_SCALAR].append(sample_value.item())
    else:
        for index, value in ndenumerate(sample_value):
            vertex_sample[index].append(value.item())


def __get_vertex_samples(network_samples, vertex) -> ndarray:
    samples_for_vertex = network_samples.get(vertex).asTensor()
    return Tensor._to_ndarray(samples_for_vertex)

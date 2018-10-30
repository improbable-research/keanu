from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vertex import Vertex
import numpy as np

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.NUTS")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.Hamiltonian")

algorithms = {'metropolis': k.jvm_view().MetropolisHastings,
              'NUTS': k.jvm_view().NUTS,
              'hamiltonian': k.jvm_view().Hamiltonian}

def sample(net, sample_from, algo='metropolis', draws=500, drop=0, down_sample_interval=1):
    vertices_unwrapped = k.to_java_object_list(sample_from)

    network_samples = algorithms[algo].withDefaultConfig().getPosteriorSamples(net.unwrap(), vertices_unwrapped, draws).drop(drop).downSample(down_sample_interval)
    vertex_samples = {Vertex._to_python_id(vertex_unwrapped.getId()): list(map(Tensor._to_ndarray, network_samples.get(vertex_unwrapped).asList())) for vertex_unwrapped in vertices_unwrapped}

    return vertex_samples

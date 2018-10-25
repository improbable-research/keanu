from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from keanu.bayes_net import BayesNet
import numpy as np

context = KeanuContext()
k = context.jvm_view()

java_import(k, "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")
java_import(k, "io.improbable.keanu.algorithms.mcmc.NUTS")
java_import(k, "io.improbable.keanu.algorithms.mcmc.Hamiltonian")

algorithms = {'metropolis': k.MetropolisHastings,
              'NUTS': k.NUTS,
              'hamiltonian': k.Hamiltonian}

def sample(net, sample_from, algo='metropolis', draws=500, drop=0, down_sample_interval=1):
    vertices = context.to_java_list([vertex.unwrap() for vertex in sample_from])

    network_samples = algorithms[algo].withDefaultConfig().getPosteriorSamples(net.unwrap(), vertices, draws).drop(drop).downSample(down_sample_interval)
    vertex_samples = {vertex.get_id(): __to_np_arrays(network_samples.get(vertex.unwrap()).asList()) for vertex in sample_from}

    return vertex_samples

def __to_np_arrays(values):
    np_arrays = []
    for value in values:
        np_array = np.array(list(value.asFlatArray()))
        np_array.reshape(value.getShape())

        np_arrays.append(np_array)
    return np_arrays

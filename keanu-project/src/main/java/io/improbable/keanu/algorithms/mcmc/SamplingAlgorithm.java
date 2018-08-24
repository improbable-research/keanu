package io.improbable.keanu.algorithms.mcmc;

import java.util.List;
import java.util.Map;

import io.improbable.keanu.network.NetworkState;

public interface SamplingAlgorithm {

    /**
     * Move forward the state of the Sampling Algorithm by a single step but do not return anything.
     */
    void step();

    /**
     * Takes a sample with the algorithm and saves it in the supplied map (creating a new entry in the list if the
     * Vertex already exists.
     *
     * @param samples map to store sampled vertex values
     */
    void sample(Map<Long, List<?>> samples);

    /**
     * Takes a sample with the algorithm and returns the state of the network for that sample.
     *
     * @return a network state that represents the current state of the algorithm.
     */
    NetworkState sample();
}

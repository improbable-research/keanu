package io.improbable.keanu.algorithms.mcmc;

import java.util.List;
import java.util.Map;

import io.improbable.keanu.network.NetworkState;

public interface SamplingAlgorithm {

    /**
     * Move forward the state of the Sampling Algorithm by a single step.
     */
    void step();

    /**
     * Takes a sample with the algorithm and saves it in the supplied map.  Repeated calls to this function will return
     * the same values without an intermediary call to 'step()'
     *
     * @param samples map to store sampled vertex values
     */
    void sample(Map<Long, List<?>> samples);

    /**
     * Takes a sample with the algorithm and returns the state of the network for that sample.  Repeated calls to this
     * function will return the same values without an intermediary call to 'step()'
     *
     * @return a network state that represents the current state of the algorithm.
     */
    NetworkState sample();
}

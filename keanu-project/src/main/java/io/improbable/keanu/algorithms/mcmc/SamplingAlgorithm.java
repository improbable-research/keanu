package io.improbable.keanu.algorithms.mcmc;

import java.util.List;
import java.util.Map;

import io.improbable.keanu.network.NetworkState;

public interface SamplingAlgorithm {

    /**
     * Same effect as a sample but the result isn't saved or returned.
     */
    void step();

    /**
     * Takes a sample with the algorithm and saves it in the supplied map
     *
     * @param samples map to store sampled vertex values
     */
    void sample(Map<Long, List<?>> samples);

    /**
     * @return a network state that represents the value of vertices at the
     * end of the algorithm step
     */
    NetworkState sample();
}

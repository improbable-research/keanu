package io.improbable.keanu.algorithms;

import java.util.List;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;

public interface PosteriorSamplingAlgorithm {

    NetworkSamples getPosteriorSamples(BayesianNetwork bayesNet,
                                       List<? extends Vertex> verticesToSampleFrom,
                                       int sampleCount);

    default NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork, int sampleCount) {
        return getPosteriorSamples(bayesianNetwork,
            bayesianNetwork.getLatentVerticesAtDepth(bayesianNetwork.getDepth()), sampleCount);
    }

}

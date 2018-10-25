package io.improbable.keanu.algorithms;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public interface PosteriorSamplingAlgorithm {

    default NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork,
                                               Vertex vertexToSampleFrom,
                                               int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, Collections.singletonList(vertexToSampleFrom), sampleCount);
    }

    NetworkSamples getPosteriorSamples(BayesianNetwork bayesNet,
                                       List<? extends Vertex> verticesToSampleFrom,
                                       int sampleCount);

    default NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork, int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, bayesianNetwork.getTopLevelLatentVertices(), sampleCount);
    }

}

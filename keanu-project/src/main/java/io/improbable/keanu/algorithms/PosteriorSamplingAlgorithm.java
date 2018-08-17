package io.improbable.keanu.algorithms;

import java.util.List;
import java.util.stream.Stream;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.Vertex;

public interface PosteriorSamplingAlgorithm {

    NetworkSamples getPosteriorSamples(BayesianNetwork bayesNet,
                                       List<? extends Vertex> verticesToSampleFrom,
                                       int sampleCount);
}

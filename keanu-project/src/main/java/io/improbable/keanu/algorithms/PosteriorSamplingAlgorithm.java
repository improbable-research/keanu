package io.improbable.keanu.algorithms;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.List;

public interface PosteriorSamplingAlgorithm {

    NetworkSamples getPosteriorSamples(final BayesianNetwork bayesNet,
                                       final List<? extends Vertex> verticesToSampleFrom,
                                       final int sampleCount,
                                       final KeanuRandom random);

    NetworkSamples getPosteriorSamples(BayesianNetwork bayesNet,
                                       List<? extends Vertex> verticesToSampleFrom,
                                       int sampleCount);
}

package io.improbable.keanu.algorithms;

import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Collections;
import java.util.List;

public interface PosteriorSamplingAlgorithm {

    default NetworkSamples getPosteriorSamples(ProbabilisticGraph bayesianNetwork,
                                               Variable vertexToSampleFrom,
                                               int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, Collections.singletonList(vertexToSampleFrom), sampleCount);
    }

    default NetworkSamples getPosteriorSamples(ProbabilisticGraph bayesianNetwork, int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, bayesianNetwork.getLatentVariables(), sampleCount);
    }

    NetworkSamples getPosteriorSamples(ProbabilisticGraph bayesNet,
                                       List<? extends Variable> verticesToSampleFrom,
                                       int sampleCount);

}

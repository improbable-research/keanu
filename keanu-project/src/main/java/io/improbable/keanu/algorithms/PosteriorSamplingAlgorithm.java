package io.improbable.keanu.algorithms;

import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Collections;
import java.util.List;

public interface PosteriorSamplingAlgorithm {

    default NetworkSamples getPosteriorSamples(ProbabilisticModel bayesianNetwork,
                                               Variable variableToSampleFrom,
                                               int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, Collections.singletonList(variableToSampleFrom), sampleCount);
    }

    default NetworkSamples getPosteriorSamples(ProbabilisticModel bayesianNetwork, int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, bayesianNetwork.getLatentVariables(), sampleCount);
    }

    NetworkSamples getPosteriorSamples(ProbabilisticModel bayesNet,
                                       List<? extends Variable> variablesToSampleFrom,
                                       int sampleCount);

}

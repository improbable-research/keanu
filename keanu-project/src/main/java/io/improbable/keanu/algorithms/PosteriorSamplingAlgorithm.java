package io.improbable.keanu.algorithms;

import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Collections;
import java.util.List;

public interface PosteriorSamplingAlgorithm {

    default NetworkSamples getPosteriorSamples(ProbabilisticModel model,
                                               Variable variableToSampleFrom,
                                               int sampleCount) {
        return getPosteriorSamples(model, Collections.singletonList(variableToSampleFrom), sampleCount);
    }

    default NetworkSamples getPosteriorSamples(ProbabilisticModel model, int sampleCount) {
        return getPosteriorSamples(model, model.getLatentVariables(), sampleCount);
    }

    NetworkSamples getPosteriorSamples(ProbabilisticModel model,
                                       List<? extends Variable> variablesToSampleFrom,
                                       int sampleCount);

}

package io.improbable.keanu.algorithms;

import io.improbable.keanu.algorithms.mcmc.NetworkSamplesGenerator;
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

    /**
     * @param model                     a model containing latent variables
     * @param variablesToSampleFrom     the variables to include in the returned samples
     * @param sampleCount               the number of samples to take
     * @return samples for each variable
     */
    NetworkSamples getPosteriorSamples(ProbabilisticModel model,
                                       List<? extends Variable> variablesToSampleFrom,
                                       int sampleCount);

    NetworkSamplesGenerator generatePosteriorSamples(final ProbabilisticModel model,
                                                     final List<? extends Variable> variableToSampleFrom);


}

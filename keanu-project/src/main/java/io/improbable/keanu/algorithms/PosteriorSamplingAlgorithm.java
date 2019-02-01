package io.improbable.keanu.algorithms;

import io.improbable.keanu.algorithms.mcmc.NetworkSamplesGenerator;
import io.improbable.keanu.vertices.RandomVariable;

import java.util.Collections;
import java.util.List;

public interface PosteriorSamplingAlgorithm {

    default NetworkSamples getPosteriorSamples(ProbabilisticModel model,
                                               RandomVariable variableToSampleFrom,
                                               int sampleCount) {
        return getPosteriorSamples(model, Collections.singletonList(variableToSampleFrom), sampleCount);
    }

    default NetworkSamples getPosteriorSamples(ProbabilisticModel model, int sampleCount) {
        return getPosteriorSamples(model, model.getRandomVariables(), sampleCount);
    }

    /**
     * @param model                     a model containing latent variables
     * @param variablesToSampleFrom     the variables to include in the returned samples
     * @param sampleCount               the number of samples to take
     * @return samples for each variable
     */
    NetworkSamples getPosteriorSamples(ProbabilisticModel model,
                                       List<? extends RandomVariable> variablesToSampleFrom,
                                       int sampleCount);

    NetworkSamplesGenerator generatePosteriorSamples(final ProbabilisticModel model,
                                                     final List<? extends RandomVariable> variableToSampleFrom);


}

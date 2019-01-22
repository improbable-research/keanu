package io.improbable.keanu.algorithms;

import io.improbable.keanu.algorithms.mcmc.NetworkSamplesGenerator;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collections;
import java.util.List;

public interface PosteriorSamplingAlgorithm {

    default NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork,
                                               Vertex vertexToSampleFrom,
                                               int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, Collections.singletonList(vertexToSampleFrom), sampleCount);
    }

    /**
     * @param bayesNet      a bayesian network containing latent vertices
     * @param verticesToSampleFrom the vertices to include in the returned samples
     * @param sampleCount          number of samples to take using the algorithm
     * @return Samples for each vertex ordered by MCMC iteration
     */
    default NetworkSamples getPosteriorSamples(BayesianNetwork bayesNet,
                                               List<? extends Vertex> verticesToSampleFrom,
                                               int sampleCount) {
        return generatePosteriorSamples(bayesNet, verticesToSampleFrom).generate(sampleCount);
    }

    default NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork, int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, bayesianNetwork.getTopLevelLatentVertices(), sampleCount);
    }

    NetworkSamplesGenerator generatePosteriorSamples(final BayesianNetwork bayesianNetwork,
                                                     final List<? extends Vertex> verticesToSampleFrom);

}

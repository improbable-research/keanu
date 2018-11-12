package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;

import java.util.List;

public class MCMC implements PosteriorSamplingAlgorithm {
    @Override
    public NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork, Vertex vertexToSampleFrom, int sampleCount) {
        if (bayesianNetwork.getNonDifferentiableVertices().isEmpty()) {
            return NUTS.withDefaultConfig().getPosteriorSamples(bayesianNetwork, vertexToSampleFrom, sampleCount);
        } else {
            return MetropolisHastings.withDefaultConfig().getPosteriorSamples(bayesianNetwork, vertexToSampleFrom, sampleCount);
        }
    }

    @Override
    public NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork, List<? extends Vertex> verticesToSampleFrom, int sampleCount) {
        if (bayesianNetwork.getNonDifferentiableVertices().isEmpty()) {
            return NUTS.withDefaultConfig().getPosteriorSamples(bayesianNetwork, verticesToSampleFrom, sampleCount);
        } else {
            return MetropolisHastings.withDefaultConfig().getPosteriorSamples(bayesianNetwork, verticesToSampleFrom, sampleCount);
        }
    }

    @Override
    public NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork, int sampleCount) {
        if (bayesianNetwork.getNonDifferentiableVertices().isEmpty()) {
            return NUTS.withDefaultConfig().getPosteriorSamples(bayesianNetwork, sampleCount);
        } else {
            return MetropolisHastings.withDefaultConfig().getPosteriorSamples(bayesianNetwork, sampleCount);
        }
    }
}

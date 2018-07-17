package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.proposals.PriorProposal;
import io.improbable.keanu.algorithms.mcmc.proposals.ProposalDistribution;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;

import java.util.*;

/**
 * Metropolis Hastings is a Markov Chain Monte Carlo method for obtaining samples from a probability distribution
 */
@Builder
public class MetropolisHastings implements PosteriorSamplingAlgorithm {

    public static MetropolisHastings withDefaultConfig() {
        return MetropolisHastings.builder()
            .proposalDistribution(PriorProposal.SINGLETON)
            .useCacheOnRejection(true)
            .build();
    }

    @Builder.Default
    private final ProposalDistribution proposalDistribution = PriorProposal.SINGLETON;

    @Builder.Default
    private final boolean useCacheOnRejection = true;

    @Override
    public NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork,
                                              List<? extends Vertex> verticesToSampleFrom,
                                              int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, verticesToSampleFrom, sampleCount, KeanuRandom.getDefaultRandom());
    }

    /**
     * @param bayesianNetwork      a bayesian network containing latent vertices
     * @param verticesToSampleFrom the vertices to include in the returned samples
     * @param sampleCount          number of samples to take using the algorithm
     * @param random               the source of randomness
     * @return Samples for each vertex ordered by MCMC iteration
     */
    @Override
    public NetworkSamples getPosteriorSamples(final BayesianNetwork bayesianNetwork,
                                              final List<? extends Vertex> verticesToSampleFrom,
                                              final int sampleCount,
                                              final KeanuRandom random) {
        checkBayesNetInHealthyState(bayesianNetwork);

        Map<Long, List<?>> samplesByVertex = new HashMap<>();
        List<Vertex> latentVertices = bayesianNetwork.getLatentVertices();
        MCMCStep mcmcStep = new MCMCStep(latentVertices, proposalDistribution, useCacheOnRejection);

        double totalLogProbability = bayesianNetwork.getLogOfMasterP();
        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {

            Vertex<?> chosenVertex = latentVertices.get(sampleNum % latentVertices.size());
            totalLogProbability = mcmcStep.nextSample(
                Collections.singleton(chosenVertex),
                totalLogProbability,
                random
            );

            takeSamples(samplesByVertex, verticesToSampleFrom);
        }

        return new NetworkSamples(samplesByVertex, sampleCount);
    }

    private static void takeSamples(Map<Long, List<?>> samples, List<? extends Vertex> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex((Vertex<?>) vertex, samples));
    }

    private static <T> void addSampleForVertex(Vertex<T> vertex, Map<Long, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<T>());
        samplesForVertex.add(vertex.getValue());
    }

    private static void checkBayesNetInHealthyState(BayesianNetwork bayesNet) {
        bayesNet.cascadeObservations();
        if (bayesNet.getLatentAndObservedVertices().isEmpty()) {
            throw new IllegalArgumentException("Cannot sample from a completely deterministic BayesNet");
        } else if (bayesNet.isInImpossibleState()) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }
    }

}

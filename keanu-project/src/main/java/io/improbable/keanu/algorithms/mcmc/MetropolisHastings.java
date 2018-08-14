package io.improbable.keanu.algorithms.mcmc;

import static io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

/**
 * Metropolis Hastings is a Markov Chain Monte Carlo method for obtaining samples from a probability distribution
 */
@Builder
public class MetropolisHastings implements PosteriorSamplingAlgorithm {

    public static MetropolisHastings withDefaultConfig() {
        return withDefaultConfig(KeanuRandom.getDefaultRandom());
    }

    public static MetropolisHastings withDefaultConfig(KeanuRandom random) {
        return MetropolisHastings.builder()
            .proposalDistribution(ProposalDistribution.usePrior())
            .variableSelector(SINGLE_VARIABLE_SELECTOR)
            .useCacheOnRejection(true)
            .random(random)
            .build();
    }

    @Getter
    @Setter
    @Builder.Default
    private KeanuRandom random = KeanuRandom.getDefaultRandom();

    @Getter
    @Setter
    @Builder.Default
    private ProposalDistribution proposalDistribution = ProposalDistribution.usePrior();

    @Getter
    @Setter
    @Builder.Default
    private MHStepVariableSelector variableSelector = SINGLE_VARIABLE_SELECTOR;

    @Getter
    @Setter
    @Builder.Default
    private boolean useCacheOnRejection = true;

    /**
     * @param bayesianNetwork      a bayesian network containing latent vertices
     * @param verticesToSampleFrom the vertices to include in the returned samples
     * @param sampleCount          number of samples to take using the algorithm
     * @return Samples for each vertex ordered by MCMC iteration
     */
    @Override
    public NetworkSamples getPosteriorSamples(final BayesianNetwork bayesianNetwork,
                                              final List<? extends Vertex> verticesToSampleFrom,
                                              final int sampleCount) {
        checkBayesNetInHealthyState(bayesianNetwork);

        Map<Long, List<?>> samplesByVertex = new HashMap<>();
        List<Vertex> latentVertices = bayesianNetwork.getLatentVertices();

        MetropolisHastingsStep mhStep = new MetropolisHastingsStep(
            latentVertices,
            proposalDistribution,
            useCacheOnRejection,
            random
        );


        double logProbabilityBeforeStep = bayesianNetwork.getLogOfMasterP();
        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {

            Set<Vertex> chosenVertices = variableSelector.select(latentVertices, sampleNum);

            logProbabilityBeforeStep = mhStep.step(
                chosenVertices,
                logProbabilityBeforeStep
            ).getLogProbabilityAfterStep();

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

package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR;

/**
 * Metropolis Hastings is a Markov Chain Monte Carlo method for obtaining samples from a probability distribution
 */
@Builder
public class MetropolisHastings implements PosteriorSamplingAlgorithm {

    private static final ProposalDistribution DEFAULT_PROPOSAL_DISTRIBUTION = ProposalDistribution.usePrior();
    private static final MHStepVariableSelector DEFAULT_VARIABLE_SELECTOR = SINGLE_VARIABLE_SELECTOR;
    private static final boolean DEFAULT_USE_CACHE_ON_REJECTION = true;

    public static MetropolisHastings withDefaultConfig() {
        return withDefaultConfig(KeanuRandom.getDefaultRandom());
    }

    public static MetropolisHastings withDefaultConfig(KeanuRandom random) {
        return MetropolisHastings.builder()
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
    private ProposalDistribution proposalDistribution = DEFAULT_PROPOSAL_DISTRIBUTION;

    @Getter
    @Setter
    @Builder.Default
    private MHStepVariableSelector variableSelector = DEFAULT_VARIABLE_SELECTOR;

    @Getter
    @Setter
    @Builder.Default
    private boolean useCacheOnRejection = DEFAULT_USE_CACHE_ON_REJECTION;

    /**
     * @param bayesianNetwork      a bayesian network containing latent vertices
     * @param verticesToSampleFrom the vertices to include in the returned samples
     * @param sampleCount          number of samples to take using the algorithm
     * @return Samples for each vertex ordered by MCMC iteration
     */
    @Override
    public NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork,
                                              List<? extends Vertex> verticesToSampleFrom,
                                              int sampleCount) {
        return generatePosteriorSamples(bayesianNetwork, verticesToSampleFrom)
            .generate(sampleCount);
    }

    public NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork,
                                              Vertex vertexToSampleFrom,
                                              int sampleCount) {
        return generatePosteriorSamples(bayesianNetwork, Collections.singletonList(vertexToSampleFrom))
            .generate(sampleCount);
    }

    public NetworkSamplesGenerator generatePosteriorSamples(final BayesianNetwork bayesianNetwork,
                                                            final List<? extends Vertex> verticesToSampleFrom) {
        return new NetworkSamplesGenerator(setupSampler(bayesianNetwork, verticesToSampleFrom), ProgressBar::new);
    }

    private SamplingAlgorithm setupSampler(final BayesianNetwork bayesianNetwork,
                                           final List<? extends Vertex> verticesToSampleFrom) {
        checkBayesNetInHealthyState(bayesianNetwork);

        List<Vertex> latentVertices = bayesianNetwork.getLatentVertices();

        MetropolisHastingsStep mhStep = new MetropolisHastingsStep(
            latentVertices,
            proposalDistribution,
            useCacheOnRejection,
            random
        );

        double logProbabilityBeforeStep = bayesianNetwork.getLogOfMasterP();

        return new Sampler(latentVertices, verticesToSampleFrom, mhStep, variableSelector, logProbabilityBeforeStep);
    }

    public static class Sampler implements SamplingAlgorithm {

        private final List<Vertex> latentVertices;
        private final List<? extends Vertex> verticesToSampleFrom;
        private final MetropolisHastingsStep mhStep;
        private final MHStepVariableSelector variableSelector;

        private double logProbabilityBeforeStep;
        private int sampleNum;

        public Sampler(List<Vertex> latentVertices,
                       List<? extends Vertex> verticesToSampleFrom,
                       MetropolisHastingsStep mhStep,
                       MHStepVariableSelector variableSelector,
                       double logProbabilityBeforeStep) {
            this.latentVertices = latentVertices;
            this.verticesToSampleFrom = verticesToSampleFrom;
            this.mhStep = mhStep;
            this.variableSelector = variableSelector;
            this.logProbabilityBeforeStep = logProbabilityBeforeStep;
            this.sampleNum = 0;
        }

        @Override
        public void step() {
            Set<Vertex> chosenVertices = variableSelector.select(latentVertices, sampleNum);

            logProbabilityBeforeStep = mhStep.step(
                chosenVertices,
                logProbabilityBeforeStep
            ).getLogProbabilityAfterStep();

            sampleNum++;
        }

        @Override
        public void sample(Map<VertexId, List<?>> samplesByVertex, List<Double> logOfMasterPForEachSample) {
            step();
            takeSamples(samplesByVertex, verticesToSampleFrom);
            logOfMasterPForEachSample.add(logProbabilityBeforeStep);
        }

        @Override
        public NetworkState sample() {
            step();
            return new SimpleNetworkState(takeSample(verticesToSampleFrom));
        }
    }

    private static Map<VertexId, ?> takeSample(List<? extends Vertex> fromVertices) {
        Map<VertexId, Object> sample = new HashMap<>();
        for (Vertex v : fromVertices) {
            sample.put(v.getId(), v.getValue());
        }
        return sample;
    }

    private static void takeSamples(Map<VertexId, List<?>> samples, List<? extends Vertex> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex((Vertex<?>) vertex, samples));
    }

    private static <T> void addSampleForVertex(Vertex<T> vertex, Map<VertexId, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<T>());
        samplesForVertex.add(vertex.getValue());
    }

    private static void checkBayesNetInHealthyState(BayesianNetwork bayesNet) {
        bayesNet.cascadeObservations();
        if (bayesNet.getLatentOrObservedVertices().isEmpty()) {
            throw new IllegalArgumentException("Cannot sample from a completely deterministic BayesNet");
        } else if (bayesNet.isInImpossibleState()) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }
    }

}

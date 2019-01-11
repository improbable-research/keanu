package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR;

/**
 * Metropolis Hastings is a Markov Chain Monte Carlo method for obtaining samples from a probability distribution
 */
@Builder
@Slf4j
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
    public NetworkSamples getPosteriorSamples(ProbabilisticGraph bayesianNetwork,
                                              List<? extends Variable> verticesToSampleFrom,
                                              int sampleCount) {
        return generatePosteriorSamples(bayesianNetwork, verticesToSampleFrom)
            .generate(sampleCount);
    }

    public NetworkSamplesGenerator generatePosteriorSamples(final ProbabilisticGraph bayesianNetwork,
                                                            final List<? extends Variable> verticesToSampleFrom) {

        return new NetworkSamplesGenerator(setupSampler(bayesianNetwork, verticesToSampleFrom), ProgressBar::new);
    }

    private SamplingAlgorithm setupSampler(final ProbabilisticGraph bayesianNetwork,
                                           final List<? extends Variable> verticesToSampleFrom) {
        checkBayesNetInHealthyState(bayesianNetwork);

        List<? extends Variable> latentVertices = bayesianNetwork.getLatentVariables();

        MetropolisHastingsStep mhStep = new MetropolisHastingsStep(
            bayesianNetwork,
            proposalDistribution,
            useCacheOnRejection,
            random
        );

        double logProbabilityBeforeStep = bayesianNetwork.logProb();

        return new MetropolisHastingsSampler(latentVertices, verticesToSampleFrom, mhStep, variableSelector, logProbabilityBeforeStep);
    }

    public static class MetropolisHastingsSampler implements SamplingAlgorithm {

        private final List<? extends Variable> latentVertices;
        private final List<? extends Variable> verticesToSampleFrom;
        private final MetropolisHastingsStep mhStep;
        private final MHStepVariableSelector variableSelector;

        private double logProbabilityBeforeStep;
        private int sampleNum;

        public MetropolisHastingsSampler(List<? extends Variable> latentVertices,
                                         List<? extends Variable> verticesToSampleFrom,
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
            Set<Variable> chosenVertices = variableSelector.select(latentVertices, sampleNum);

            logProbabilityBeforeStep = mhStep.step(
                chosenVertices,
                logProbabilityBeforeStep
            ).getLogProbabilityAfterStep();

            sampleNum++;
        }

        @Override
        public void sample(Map<VariableReference, List<?>> samplesByVertex, List<Double> logOfMasterPForEachSample) {
        }

        public void sampleLegacy(Map<VariableReference, List<?>> samplesByVertex, List<Double> logOfMasterPForEachSample) {
            step();
            takeSamples(samplesByVertex, verticesToSampleFrom);
            logOfMasterPForEachSample.add(logProbabilityBeforeStep);
        }

        @Override
        public NetworkSample sample() {
            step();
            return new NetworkSample(SamplingAlgorithm.takeSample((List<? extends Variable<Object>>) verticesToSampleFrom), logProbabilityBeforeStep);
        }
    }

    private static void takeSamples(Map<VariableReference, List<?>> samples, List<? extends Variable> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex((Variable<?>) vertex, samples));
    }

    private static <T> void addSampleForVertex(Variable<T> vertex, Map<VariableReference, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertex.getReference(), v -> new ArrayList<T>());
        T value = vertex.getValue();
        samplesForVertex.add(value);
        log.trace(String.format("Sampled %s", value));
    }

    private static void checkBayesNetInHealthyState(ProbabilisticGraph bayesNet) {
        bayesNet.cascadeFixedVariables();
        if (bayesNet.isDeterministic()) {
            throw new IllegalArgumentException("Cannot sample from a completely deterministic BayesNet");
        } else if (ProbabilityCalculator.isImpossibleLogProb(bayesNet.logProb())) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }
    }

}

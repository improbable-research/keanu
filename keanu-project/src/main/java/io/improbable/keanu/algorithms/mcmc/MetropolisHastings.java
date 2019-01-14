package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

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


    private static void checkBayesNetInHealthyState(ProbabilisticGraph bayesNet) {
        bayesNet.cascadeFixedVariables();
        if (bayesNet.isDeterministic()) {
            throw new IllegalArgumentException("Cannot sample from a completely deterministic BayesNet");
        } else if (ProbabilityCalculator.isImpossibleLogProb(bayesNet.logProb())) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }
    }

}

package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

import static io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR;

/**
 * Metropolis Hastings is a Markov Chain Monte Carlo method for obtaining samples from a probability distribution
 */
@Builder
public class MetropolisHastings implements PosteriorSamplingAlgorithm {

    private static final ProposalDistribution DEFAULT_PROPOSAL_DISTRIBUTION = ProposalDistribution.usePrior();
    private static final MHStepVariableSelector DEFAULT_VARIABLE_SELECTOR = SINGLE_VARIABLE_SELECTOR;
    public static final CascadeOnRejection DEFAULT_REJECTION_STRATEGY = new CascadeOnRejection();
    private static final LogProbCalculationStrategy DEFAULT_LOG_PROB_CALCULATION_STRATEGY = new SimpleLogProbCalculationStrategy();

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
    private ProposalRejectionStrategy rejectionStrategy = DEFAULT_REJECTION_STRATEGY;

    @Getter
    @Setter
    @Builder.Default
    private LogProbCalculationStrategy logProbCalculationStrategy = DEFAULT_LOG_PROB_CALCULATION_STRATEGY;

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

    public NetworkSamplesGenerator generatePosteriorSamples(final BayesianNetwork bayesianNetwork,
                                                            final List<? extends Variable> verticesToSampleFrom) {

        return new NetworkSamplesGenerator(setupSampler(new KeanuProbabilisticGraph(bayesianNetwork), verticesToSampleFrom), ProgressBar::new);
    }

    private SamplingAlgorithm setupSampler(final ProbabilisticGraph bayesianNetwork,
                                           final List<? extends Variable> verticesToSampleFrom) {

        MetropolisHastingsStep mhStep = new MetropolisHastingsStep(
            bayesianNetwork,
            proposalDistribution,
            rejectionStrategy,
            logProbCalculationStrategy,
            random
        );

        return bayesianNetwork.metropolisHastingsSampler(verticesToSampleFrom, mhStep, variableSelector);
    }

}

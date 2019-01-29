package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.util.status.StatusBar;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Getter;
import lombok.NonNull;

import java.util.List;

import static io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR;

/**
 * Metropolis Hastings is a Markov Chain Monte Carlo method for obtaining samples from a probability distribution
 */
public class MetropolisHastings implements PosteriorSamplingAlgorithm {

    private static final MHStepVariableSelector DEFAULT_VARIABLE_SELECTOR = SINGLE_VARIABLE_SELECTOR;

    @Getter
    private KeanuRandom random;

    @Getter
    @NonNull
    private ProposalDistribution proposalDistribution;

    @Getter
    private MHStepVariableSelector variableSelector;

    @Getter
    @NonNull
    private ProposalRejectionStrategy rejectionStrategy;

    @java.beans.ConstructorProperties({"random", "proposalDistribution", "variableSelector", "rejectionStrategy"})
    MetropolisHastings(KeanuRandom random, ProposalDistribution proposalDistribution, MHStepVariableSelector variableSelector, ProposalRejectionStrategy rejectionStrategy) {
        this.random = random;
        this.proposalDistribution = proposalDistribution;
        this.variableSelector = variableSelector;
        this.rejectionStrategy = rejectionStrategy;
    }

    public static MetropolisHastingsBuilder builder() {
        return new MetropolisHastingsBuilder();
    }

    /**
     * @param model      a probabilistic model containing latent variables
     * @param variablesToSampleFrom the variables to include in the returned samples
     * @param sampleCount          number of samples to take using the algorithm
     * @return Samples for each variable ordered by MCMC iteration
     */
    @Override
    public NetworkSamples getPosteriorSamples(ProbabilisticModel model,
                                              List<? extends Variable> variablesToSampleFrom,
                                              int sampleCount) {
        return generatePosteriorSamples(model, variablesToSampleFrom)
            .generate(sampleCount);
    }

    @Override
    public NetworkSamplesGenerator generatePosteriorSamples(final ProbabilisticModel model,
                                                            final List<? extends Variable> variablesToSampleFrom) {

        return new NetworkSamplesGenerator(setupSampler(model, variablesToSampleFrom), StatusBar::new);
    }

    private SamplingAlgorithm setupSampler(final ProbabilisticModel model,
                                           final List<? extends Variable> variablesToSampleFrom) {

        MetropolisHastingsStep mhStep = new MetropolisHastingsStep(
            model,
            proposalDistribution,
            rejectionStrategy,
            random
        );

        return new MetropolisHastingsSampler(model.getLatentVariables(), variablesToSampleFrom, mhStep, variableSelector, model.logProb());
    }

    public static class MetropolisHastingsBuilder {
        private KeanuRandom random = KeanuRandom.getDefaultRandom();
        private ProposalDistribution proposalDistribution;
        private MHStepVariableSelector variableSelector = DEFAULT_VARIABLE_SELECTOR;
        private ProposalRejectionStrategy rejectionStrategy;

        MetropolisHastingsBuilder() {
        }

        public MetropolisHastingsBuilder random(KeanuRandom random) {
            this.random = random;
            return this;
        }

        public MetropolisHastingsBuilder proposalDistribution(ProposalDistribution proposalDistribution) {
            this.proposalDistribution = proposalDistribution;
            return this;
        }

        public MetropolisHastingsBuilder variableSelector(MHStepVariableSelector variableSelector) {
            this.variableSelector = variableSelector;
            return this;
        }

        public MetropolisHastingsBuilder rejectionStrategy(ProposalRejectionStrategy rejectionStrategy) {
            this.rejectionStrategy = rejectionStrategy;
            return this;
        }

        public MetropolisHastings build() {
            return new MetropolisHastings(random, proposalDistribution, variableSelector, rejectionStrategy);
        }

        public String toString() {
            return "MetropolisHastings.MetropolisHastingsBuilder(random=" + this.random + ", proposalDistribution=" + this.proposalDistribution + ", variableSelector=" + this.variableSelector + ", rejectionStrategy=" + this.rejectionStrategy + ")";
        }
    }
}

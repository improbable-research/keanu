package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.proposal.GaussianProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;

import java.util.List;

import static io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR;

/**
 * Metropolis Hastings is a Markov Chain Monte Carlo method for obtaining samples from a probability distribution
 */
@Builder
public class MetropolisHastings implements PosteriorSamplingAlgorithm {

    private static final MHStepVariableSelector DEFAULT_VARIABLE_SELECTOR = SINGLE_VARIABLE_SELECTOR;

    @Getter
    @Builder.Default
    private KeanuRandom random = KeanuRandom.getDefaultRandom();

    @Getter
    @NonNull
    private ProposalDistribution proposalDistribution;

    @Getter
    @Builder.Default
    private MHStepVariableSelector variableSelector = DEFAULT_VARIABLE_SELECTOR;

    @Getter
    @NonNull
    private ProposalRejectionStrategy rejectionStrategy;

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

        return new NetworkSamplesGenerator(setupSampler(model, variablesToSampleFrom), ProgressBar::new);
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

}

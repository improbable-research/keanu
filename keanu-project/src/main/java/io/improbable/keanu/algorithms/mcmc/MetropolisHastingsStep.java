package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;

import java.util.Set;

@Slf4j
public class MetropolisHastingsStep {

    //Temperature for standard MH step accept/reject calculation
    private static final double DEFAULT_TEMPERATURE = 1.0;

    private final ProbabilisticModel model;
    private final ProposalDistribution proposalDistribution;
    private final ProposalRejectionStrategy rejectionStrategy;
    private final LogProbCalculationStrategy logProbCalculationStrategy;
    private final ProposalApplicationStrategy proposalApplicationStrategy;
    private final KeanuRandom random;

    /**
     * @param proposalDistribution        The proposal distribution
     * @param rejectionStrategy           What to do when a proposal is rejected.
     *                                    Options include {@link CascadeOnRejection} and {@link RollBackOnRejection}.
     * @param logProbCalculationStrategy  How to calculate log probability.
     *                                    Options include {@link SimpleLogProbCalculationStrategy} and {@link LambdaSectionOptimizedLogProbCalculator}
     * @param proposalApplicationStrategy What to do when a proposal is applied.
     *                                    Options include {@link CascadeOnApplication} and {@link NoActionOnApplication}
     * @param random                      Source of randomness
     */
    MetropolisHastingsStep(ProbabilisticModel model,
                           ProposalDistribution proposalDistribution,
                           ProposalRejectionStrategy rejectionStrategy,
                           LogProbCalculationStrategy logProbCalculationStrategy,
                           ProposalApplicationStrategy proposalApplicationStrategy,
                           KeanuRandom random) {

        this.model = model;
        this.proposalDistribution = proposalDistribution;
        this.rejectionStrategy = rejectionStrategy;
        this.logProbCalculationStrategy = logProbCalculationStrategy;
        this.proposalApplicationStrategy = proposalApplicationStrategy;
        this.random = random;
    }

    public StepResult step(final Set<Variable> chosenVariables,
                           final double logProbabilityBeforeStep) {
        return step(chosenVariables, logProbabilityBeforeStep, DEFAULT_TEMPERATURE);
    }

    /**
     * @param chosenVariables          variables to get a proposed change for
     * @param logProbabilityBeforeStep The log of the previous state's probability
     * @param temperature              Temperature for simulated annealing. This
     *                                 should be constant if no annealing is wanted
     * @return the log probability of the network after either accepting or rejecting the sample
     */
    public StepResult step(final Set<Variable> chosenVariables,
                           final double logProbabilityBeforeStep,
                           final double temperature) {

        final double affectedVariablesLogProbOld = logProbCalculationStrategy.calculate(model, chosenVariables);

        rejectionStrategy.prepare(chosenVariables);

        Proposal proposal = proposalDistribution.getProposal(chosenVariables, random);
        proposal.apply();
        proposalApplicationStrategy.apply(proposal, chosenVariables);

        final double affectedVariablesLogProbNew = logProbCalculationStrategy.calculate(model, chosenVariables);

        if (!ProbabilityCalculator.isImpossibleLogProb(affectedVariablesLogProbNew)) {

            final double logProbabilityDelta = affectedVariablesLogProbNew - affectedVariablesLogProbOld;
            final double logProbabilityAfterStep = logProbabilityBeforeStep + logProbabilityDelta;

            final double pqxOld = proposalDistribution.logProbAtFromGivenTo(proposal);
            final double pqxNew = proposalDistribution.logProbAtToGivenFrom(proposal);

            final double annealFactor = (1.0 / temperature);
            final double hastingsCorrection = pqxOld - pqxNew;
            final double logR = annealFactor * logProbabilityDelta + hastingsCorrection;
            final double r = Math.exp(logR);

            final boolean shouldAccept = r >= random.nextDouble();

            if (shouldAccept) {
                return new StepResult(true, logProbabilityAfterStep);
            }
        }

        proposal.reject();

        rejectionStrategy.handle();

        return new StepResult(false, logProbabilityBeforeStep);
    }

    @Value
    static class StepResult {
        boolean accepted;
        double logProbabilityAfterStep;
    }

}

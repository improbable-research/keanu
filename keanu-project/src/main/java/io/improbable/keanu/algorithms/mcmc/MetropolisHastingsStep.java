package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;

import java.util.Set;
import java.util.stream.Collectors;

@Slf4j
public class MetropolisHastingsStep {

    //Temperature for standard MH step accept/reject calculation
    private static final double DEFAULT_TEMPERATURE = 1.0;

    private final ProbabilisticGraph graph;
    private final ProposalDistribution proposalDistribution;
    private final ProposalRejectionStrategy rejectionStrategy;
    private final KeanuRandom random;

    /**
     * @param proposalDistribution The proposal distribution
     * @param rejectionStrategy What to do when a proposal is rejected.
     *                         Options include {@link CascadeOnRejection} and {@link RollBackOnRejection}.
     * @param random               Source of randomness
     */
    MetropolisHastingsStep(ProbabilisticGraph graph,
                           ProposalDistribution proposalDistribution,
                           ProposalRejectionStrategy rejectionStrategy,
                           KeanuRandom random) {

        this.graph = graph;
        this.proposalDistribution = proposalDistribution;
        this.rejectionStrategy = rejectionStrategy;
        this.random = random;
    }

    public StepResult step(final Set<Variable> chosenVertices,
                           final double logProbabilityBeforeStep) {
        return step(chosenVertices, logProbabilityBeforeStep, DEFAULT_TEMPERATURE);
    }

    /**
     * @param chosenVertices           vertices to get a proposed change for
     * @param logProbabilityBeforeStep The log of the previous state's probability
     * @param temperature              Temperature for simulated annealing. This
     *                                 should be constant if no annealing is wanted
     * @return the log probability of the network after either accepting or rejecting the sample
     */
    public StepResult step(final Set<Variable> chosenVertices,
                           final double logProbabilityBeforeStep,
                           final double temperature) {

        log.trace(String.format("Chosen vertices: %s", chosenVertices.stream()
            .map(Variable::toString)
            .collect(Collectors.toList())));
        final double affectedVerticesLogProbOld = graph.downstreamLogProb(chosenVertices);

        rejectionStrategy.prepare(chosenVertices);

        Proposal proposal = proposalDistribution.getProposal(chosenVertices, random);
        proposal.apply();
        graph.cascadeUpdate(chosenVertices);

        final double affectedVerticesLogProbNew = graph.downstreamLogProb(chosenVertices);

        if (!ProbabilityCalculator.isImpossibleLogProb(affectedVerticesLogProbNew)) {

            final double logProbabilityDelta = affectedVerticesLogProbNew - affectedVerticesLogProbOld;
            final double logProbabilityAfterStep = logProbabilityBeforeStep + logProbabilityDelta;

            final double pqxOld = proposalDistribution.logProbAtFromGivenTo(proposal);
            final double pqxNew = proposalDistribution.logProbAtToGivenFrom(proposal);

            final double annealFactor = (1.0 / temperature);
            final double hastingsCorrection = pqxOld - pqxNew;
            final double logR = annealFactor * logProbabilityDelta + hastingsCorrection;
            final double r = Math.exp(logR);

            final boolean shouldAccept = r >= random.nextDouble();

            if (shouldAccept) {
                log.trace(String.format("ACCEPT %.4f", logR));
                log.trace(String.format("New log prob = %.4f", logProbabilityAfterStep));
                return new StepResult(true, logProbabilityAfterStep);
            } else {
                log.trace(String.format("REJECT %.4f", logR));
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

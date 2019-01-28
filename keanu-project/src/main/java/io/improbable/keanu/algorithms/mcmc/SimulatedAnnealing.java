package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Getter;
import lombok.NonNull;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR;

/**
 * Simulated Annealing is a modified version of Metropolis Hastings that causes the MCMC random walk to
 * tend towards the Maximum A Posteriori (MAP)
 */
public class SimulatedAnnealing {

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
    SimulatedAnnealing(KeanuRandom random, ProposalDistribution proposalDistribution, MHStepVariableSelector variableSelector, ProposalRejectionStrategy rejectionStrategy) {
        this.random = random;
        this.proposalDistribution = proposalDistribution;
        this.variableSelector = variableSelector;
        this.rejectionStrategy = rejectionStrategy;
    }

    public static SimulatedAnnealingBuilder builder() {
        return new SimulatedAnnealingBuilder();
    }

    public NetworkState getMaxAPosteriori(ProbabilisticModel model,
                                          int sampleCount) {
        AnnealingSchedule schedule = exponentialSchedule(sampleCount, 2, 0.01);
        return getMaxAPosteriori(model, sampleCount, schedule);
    }

    /**
     * Finds the MAP using the default annealing schedule, which is an exponential decay schedule.
     *
     * @param model          a probabilistic model containing latent variables
     * @param sampleCount       the number of samples to take
     * @param annealingSchedule the schedule to update T (temperature) as a function of sample number.
     * @return the NetworkState that represents the Max A Posteriori
     */
    public NetworkState getMaxAPosteriori(ProbabilisticModel model,
                                          int sampleCount,
                                          AnnealingSchedule annealingSchedule) {

        if (ProbabilityCalculator.isImpossibleLogProb(model.logProb())) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        Map<VariableReference, ?> maxSamplesByVariable = new HashMap<>();
        List<? extends Variable> latentVariables = model.getLatentVariables();

        double logProbabilityBeforeStep = model.logProb();
        double maxLogP = logProbabilityBeforeStep;
        setSamplesAsMax(maxSamplesByVariable, latentVariables);


        MetropolisHastingsStep mhStep = new MetropolisHastingsStep(
            model,
            proposalDistribution,
            rejectionStrategy,
            random
        );

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {

            Variable<?, ?> chosenVariable = latentVariables.get(sampleNum % latentVariables.size());

            double temperature = annealingSchedule.getTemperature(sampleNum);
            logProbabilityBeforeStep = mhStep.step(
                Collections.singleton(chosenVariable),
                logProbabilityBeforeStep,
                temperature
            ).getLogProbabilityAfterStep();

            if (logProbabilityBeforeStep > maxLogP) {
                maxLogP = logProbabilityBeforeStep;
                setSamplesAsMax(maxSamplesByVariable, latentVariables);
            }
        }

        return new SimpleNetworkState(maxSamplesByVariable);
    }

    private static void setSamplesAsMax(Map<VariableReference, ?> samples, List<? extends Variable> fromVariables) {
        fromVariables.forEach(variable -> setSampleForVariable((Variable<?, ?>) variable, samples));
    }

    private static <T> void setSampleForVariable(Variable<T, ?> variable, Map<VariableReference, ?> samples) {
        ((Map<VariableReference, ? super T>) samples).put(variable.getReference(), variable.getValue());
    }

    /**
     * An annealing schedule determines how T (temperature) changes as
     * a function of the current iteration number (i.e. sample number)
     */
    public interface AnnealingSchedule {
        double getTemperature(int iteration);
    }

    /**
     * @param iterations the number of iterations annealing over
     * @param startT     the value of T at iteration 0
     * @param endT       the value of T at the last iteration
     * @return the annealing schedule
     */
    public static AnnealingSchedule exponentialSchedule(int iterations, double startT, double endT) {

        final double minusK = Math.log(endT / startT) / iterations;

        return n -> startT * Math.exp(minusK * n);
    }

    public static class SimulatedAnnealingBuilder {
        private KeanuRandom random = KeanuRandom.getDefaultRandom();
        private ProposalDistribution proposalDistribution;
        private MHStepVariableSelector variableSelector = DEFAULT_VARIABLE_SELECTOR;
        private ProposalRejectionStrategy rejectionStrategy;

        SimulatedAnnealingBuilder() {
        }

        public SimulatedAnnealingBuilder random(KeanuRandom random) {
            this.random = random;
            return this;
        }

        public SimulatedAnnealingBuilder proposalDistribution(ProposalDistribution proposalDistribution) {
            this.proposalDistribution = proposalDistribution;
            return this;
        }

        public SimulatedAnnealingBuilder variableSelector(MHStepVariableSelector variableSelector) {
            this.variableSelector = variableSelector;
            return this;
        }

        public SimulatedAnnealingBuilder rejectionStrategy(ProposalRejectionStrategy rejectionStrategy) {
            this.rejectionStrategy = rejectionStrategy;
            return this;
        }

        public SimulatedAnnealing build() {
            return new SimulatedAnnealing(random, proposalDistribution, variableSelector, rejectionStrategy);
        }

        public String toString() {
            return "SimulatedAnnealing.SimulatedAnnealingBuilder(random=" + this.random + ", proposalDistribution=" + this.proposalDistribution + ", variableSelector=" + this.variableSelector + ", rejectionStrategy=" + this.rejectionStrategy + ")";
        }
    }
}

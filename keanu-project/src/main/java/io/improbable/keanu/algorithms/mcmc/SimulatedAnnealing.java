package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR;

/**
 * Simulated Annealing is a modified version of Metropolis Hastings that causes the MCMC random walk to
 * tend towards the Maximum A Posteriori (MAP)
 */
@Builder
public class SimulatedAnnealing {

    private static final ProposalDistribution DEFAULT_PROPOSAL_DISTRIBUTION = ProposalDistribution.usePrior();
    private static final MHStepVariableSelector DEFAULT_VARIABLE_SELECTOR = SINGLE_VARIABLE_SELECTOR;
    private static final boolean DEFAULT_USE_CACHE_ON_REJECTION = true;

    public static SimulatedAnnealing withDefaultConfig() {
        return withDefaultConfig(KeanuRandom.getDefaultRandom());
    }

    public static SimulatedAnnealing withDefaultConfig(KeanuRandom random) {
        return SimulatedAnnealing.builder()
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

    public NetworkState getMaxAPosteriori(ProbabilisticGraph bayesNet,
                                          int sampleCount) {
        AnnealingSchedule schedule = exponentialSchedule(sampleCount, 2, 0.01);
        return getMaxAPosteriori(bayesNet, sampleCount, schedule);
    }

    /**
     * Finds the MAP using the default annealing schedule, which is an exponential decay schedule.
     *
     * @param bayesNet          a bayesian network containing latent vertices
     * @param sampleCount       the number of samples to take
     * @param annealingSchedule the schedule to update T (temperature) as a function of sample number.
     * @return the NetworkState that represents the Max A Posteriori
     */
    public NetworkState getMaxAPosteriori(ProbabilisticGraph bayesNet,
                                          int sampleCount,
                                          AnnealingSchedule annealingSchedule) {

        if (ProbabilityCalculator.isImpossibleLogProb(bayesNet.logProb())) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        Map<VariableReference, ?> maxSamplesByVertex = new HashMap<>();
        List<? extends Variable> latentVariables = bayesNet.getLatentVariables();
        List<Vertex> latentVertices = (List<Vertex>) latentVariables;

        double logProbabilityBeforeStep = bayesNet.logProb();
        double maxLogP = logProbabilityBeforeStep;
        setSamplesAsMax(maxSamplesByVertex, latentVariables);


        MetropolisHastingsStep mhStep = new MetropolisHastingsStep(
            bayesNet,
            proposalDistribution,
            new RollBackOnRejection(latentVertices),
            new LambdaSectionOptimizedLogProbCalculator(latentVertices),
            new CascadingApplicationStrategy(),
            random
        );

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {

            Variable<?> chosenVertex = latentVariables.get(sampleNum % latentVariables.size());

            double temperature = annealingSchedule.getTemperature(sampleNum);
            logProbabilityBeforeStep = mhStep.step(
                Collections.singleton(chosenVertex),
                logProbabilityBeforeStep,
                temperature
            ).getLogProbabilityAfterStep();

            if (logProbabilityBeforeStep > maxLogP) {
                maxLogP = logProbabilityBeforeStep;
                setSamplesAsMax(maxSamplesByVertex, latentVariables);
            }
        }

        return new SimpleNetworkState(mapByVariableReference(maxSamplesByVertex));
    }

    private <T> Map<VariableReference, T> mapByVariableReference(Map<VariableReference, T> legacyMap) {
        return legacyMap.entrySet().stream().collect(Collectors.toMap(k -> (VariableReference) k.getKey(), Map.Entry::getValue));
    }

    private static void setSamplesAsMax(Map<VariableReference, ?> samples, List<? extends Variable> fromVertices) {
        fromVertices.forEach(vertex -> setSampleForVertex((Variable<?>) vertex, samples));
    }

    private static <T> void setSampleForVertex(Variable<T> vertex, Map<VariableReference, ?> samples) {
        ((Map<VariableReference, ? super T>) samples).put(vertex.getReference(), vertex.getValue());
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

}

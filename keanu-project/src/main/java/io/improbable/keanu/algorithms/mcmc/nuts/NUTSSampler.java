package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.mcmc.SamplingAlgorithm;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Algorithm 6: "No-U-Turn Sampler with Dual Averaging".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
@Slf4j
class NUTSSampler implements SamplingAlgorithm {

    private final List<? extends Variable> sampleFromVariables;
    private final ProbabilisticModelWithGradient logProbGradientCalculator;
    private final LeapfrogIntegrator leapfrogIntegrator;

    private final boolean adaptPotentialEnabled;
    private final Potential potential;

    private final boolean adaptStepSizeEnabled;
    private final AdaptiveStepSize stepSize;

    private final long adaptCount;

    private long stepCount;

    private final int maxTreeHeight;

    private Proposal proposal;

    private final KeanuRandom random;

    private final double maxEnergyChange;

    private final Statistics statistics;
    private final boolean saveStatistics;


    /**
     * @param sampleFromVariables       variables to sample from
     * @param logProbGradientCalculator gradient calculator for diff of log prob with respect to latents
     * @param adaptPotentialEnabled     enable the potential adaption
     * @param potential                 provides mass in velocity and energy calculations
     * @param adaptStepSizeEnabled      enable the NUTS step size adaptation
     * @param stepSize                  configuration for tuning the stepSize, if adaptStepSizeEnabled
     * @param adaptCount                number of steps to adapt potential and step size
     * @param maxEnergyChange           the maximum change in energy before a step is considered divergent
     * @param maxTreeHeight             The largest tree height before stopping the Hamiltonian process
     * @param initialProposal           the starting proposal for the tree
     * @param random                    the source of randomness
     * @param statistics                the sampler statistics
     * @param saveStatistics            whether to record statistics
     */
    public NUTSSampler(List<? extends Variable> sampleFromVariables,
                       ProbabilisticModelWithGradient logProbGradientCalculator,
                       boolean adaptPotentialEnabled,
                       Potential potential,
                       boolean adaptStepSizeEnabled,
                       AdaptiveStepSize stepSize,
                       long adaptCount,
                       double maxEnergyChange,
                       int maxTreeHeight,
                       Proposal initialProposal,
                       KeanuRandom random,
                       Statistics statistics,
                       boolean saveStatistics) {

        this.sampleFromVariables = sampleFromVariables;
        this.logProbGradientCalculator = logProbGradientCalculator;
        this.leapfrogIntegrator = new LeapfrogIntegrator(potential);

        this.adaptPotentialEnabled = adaptPotentialEnabled;
        this.potential = potential;

        this.adaptStepSizeEnabled = adaptStepSizeEnabled;
        this.stepSize = stepSize;

        this.adaptCount = adaptCount;
        this.stepCount = 0;

        this.maxEnergyChange = maxEnergyChange;
        this.maxTreeHeight = maxTreeHeight;

        this.proposal = initialProposal;

        this.random = random;
        this.statistics = statistics;
        this.saveStatistics = saveStatistics;

    }

    @Override
    public void sample(Map<VariableReference, List<?>> samples, List<Double> logOfMasterPForEachSample) {
        step();
        addSampleFromCache(samples, proposal.getSample());
        logOfMasterPForEachSample.add(proposal.getLogProb());
    }

    @Override
    public NetworkSample sample() {
        step();
        return new NetworkSample(proposal.getSample(), proposal.getLogProb());
    }

    @Override
    public void step() {

        Map<VariableReference, DoubleTensor> initialMomentum = potential.randomMomentum(random);

        LeapfrogState startState = new LeapfrogState(
            proposal.getPosition(),
            initialMomentum,
            proposal.getGradient(),
            proposal.getLogProb(),
            potential
        );

        Tree tree = new Tree(
            startState,
            proposal,
            maxEnergyChange,
            logProbGradientCalculator,
            leapfrogIntegrator,
            sampleFromVariables,
            random
        );


        while (tree.shouldContinue() && tree.getTreeHeight() < maxTreeHeight) {

            //build tree direction -1 = backwards OR 1 = forwards
            int buildDirection = random.nextBoolean() ? 1 : -1;

            tree.grow(buildDirection, stepSize.getStepSize());
        }

        this.proposal = tree.getProposal();

        if (saveStatistics) {
            stepSize.save(statistics);
            tree.save(statistics);
        }

        if (this.adaptStepSizeEnabled) {
            stepSize.adaptStepSize(tree);
        }

        if (stepCount < adaptCount && this.adaptPotentialEnabled) {
            potential.update(proposal.getPosition());
        }

        if (stepCount > adaptCount) {
            if (tree.isDiverged()) {
                statistics.store(NUTS.Metrics.DIVERGENT_SAMPLE, (double) stepCount);
                log.warn("Divergent NUTS sample after adaption ended. Increase the number or samples to adapt for or the max energy change.");
            }
        }

        stepCount++;
    }

    /**
     * This is used to save of the sample from the uniformly chosen acceptedPosition position
     *
     * @param samples      samples taken already
     * @param cachedSample a cached sample from before leapfrog
     */
    private static void addSampleFromCache(Map<VariableReference, List<?>> samples, Map<VariableReference, ?> cachedSample) {
        for (Map.Entry<VariableReference, ?> sampleEntry : cachedSample.entrySet()) {
            addSampleForVariable(sampleEntry.getKey(), sampleEntry.getValue(), samples);
        }
    }

    private static <T> void addSampleForVariable(VariableReference id, T value, Map<VariableReference, List<?>> samples) {
        List<T> samplesForVariable = (List<T>) samples.computeIfAbsent(id, v -> new ArrayList<T>());
        samplesForVariable.add(value);
    }

}

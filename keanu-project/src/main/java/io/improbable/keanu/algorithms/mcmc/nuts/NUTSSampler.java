package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.mcmc.SamplingAlgorithm;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Algorithm 6: "No-U-Turn Sampler with Dual Averaging".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
class NUTSSampler implements SamplingAlgorithm {

    private final KeanuRandom random;
    private final List<? extends Variable> sampleFromVariables;
    private final int maxTreeHeight;
    private final boolean adaptEnabled;
    private final AdaptiveStepSize stepSize;
    private final double maxEnergyChange;
    private final ProbabilisticModelWithGradient logProbGradientCalculator;
    private final Statistics statistics;
    private final boolean saveStatistics;
    private final Potential potential;

    private Proposal proposal;
    private int sampleNum;

    /**
     * @param sampleFromVariables       variables to sample from
     * @param logProbGradientCalculator gradient calculator for diff of log prob with respect to latents
     * @param potential                 ????
     * @param adaptEnabled              enable the NUTS step size adaptation
     * @param stepSize                  configuration for tuning the stepSize, if adaptEnabled
     * @param initialProposal           the starting proposal for the tree
     * @param maxTreeHeight             The largest tree height before stopping the hamilitonian process
     * @param random                    the source of randomness
     * @param statistics                the sampler statistics
     * @param saveStatistics            whether to record statistics
     */
    public NUTSSampler(List<? extends Variable> sampleFromVariables,
                       ProbabilisticModelWithGradient logProbGradientCalculator,
                       Potential potential,
                       boolean adaptEnabled,
                       AdaptiveStepSize stepSize,
                       double maxEnergyChange,
                       Proposal initialProposal,
                       int maxTreeHeight,
                       KeanuRandom random,
                       Statistics statistics,
                       boolean saveStatistics) {

        this.sampleFromVariables = sampleFromVariables;
        this.logProbGradientCalculator = logProbGradientCalculator;

        this.proposal = initialProposal;
        this.stepSize = stepSize;
        this.maxEnergyChange = maxEnergyChange;
        this.maxTreeHeight = maxTreeHeight;
        this.adaptEnabled = adaptEnabled;

        this.random = random;
        this.statistics = statistics;
        this.saveStatistics = saveStatistics;

        this.sampleNum = 1;

        this.potential = potential;
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

        Map<VariableReference, DoubleTensor> initialMomentum = potential.random();

        Leapfrog startState = new Leapfrog(
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
            sampleFromVariables,
            random
        );

        int treeHeight = 0;

        while (tree.shouldContinue() && treeHeight < maxTreeHeight) {

            //build tree direction -1 = backwards OR 1 = forwards
            int buildDirection = random.nextBoolean() ? 1 : -1;

            tree.growTree(treeHeight, buildDirection, stepSize.getStepSize());

            treeHeight++;
        }

        this.proposal = tree.getProposal();

        if (saveStatistics) {
            stepSize.save(statistics);
            tree.save(statistics);
        }

        if (this.adaptEnabled) {
            stepSize.adaptStepSize(tree);
            potential.update(proposal.getPosition(), proposal.getGradient(), sampleNum);
        }

        sampleNum++;
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

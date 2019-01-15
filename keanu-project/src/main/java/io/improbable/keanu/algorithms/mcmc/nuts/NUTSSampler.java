package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.mcmc.SamplingAlgorithm;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

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
    private final List<? extends Variable<DoubleTensor>> latentVariables;
    private final List<? extends Variable> sampleFromVariables;
    private final int maxTreeHeight;
    private final boolean adaptEnabled;
    private final Stepsize stepsize;
    private final Tree tree;
    private final ProbabilisticModelWithGradient logProbGradientCalculator;
    private final Statistics statistics;
    private final boolean saveStatistics;
    private int sampleNum;

    /**
     * @param sampleFromVariables        variables to sample from
     * @param latentVariables            variables that represent latent variables
     * @param logProbGradientCalculator gradient calculator for diff of log prob with respect to latents
     * @param adaptEnabled              enable the NUTS step size adaptation
     * @param stepsize                  configuration for tuning the stepsize, if adaptEnabled
     * @param tree                      initial tree that will contain the state of the tree build
     * @param maxTreeHeight             The largest tree height before stopping the hamilitonian process
     * @param random                    the source of randomness
     * @param statistics                the sampler statistics
     * @param saveStatistics            whether to record statistics
     */
    public NUTSSampler(List<? extends Variable> sampleFromVariables,
                       List<? extends Variable<DoubleTensor>> latentVariables,
                       ProbabilisticModelWithGradient logProbGradientCalculator,
                       boolean adaptEnabled,
                       Stepsize stepsize,
                       Tree tree,
                       int maxTreeHeight,
                       KeanuRandom random,
                       Statistics statistics,
                       boolean saveStatistics) {

        this.sampleFromVariables = sampleFromVariables;
        this.latentVariables = latentVariables;
        this.logProbGradientCalculator = logProbGradientCalculator;

        this.tree = tree;
        this.stepsize = stepsize;
        this.maxTreeHeight = maxTreeHeight;
        this.adaptEnabled = adaptEnabled;

        this.random = random;
        this.statistics = statistics;
        this.saveStatistics = saveStatistics;

        this.sampleNum = 1;
    }

    @Override
    public void sample(Map<VariableReference, List<?>> samples, List<Double> logOfMasterPForEachSample) {
        step();
        addSampleFromCache(samples, tree.getSampleAtAcceptedPosition());
        logOfMasterPForEachSample.add(tree.getLogOfMasterPAtAcceptedPosition());
    }

    @Override
    public NetworkSample sample() {
        step();
        return new NetworkSample(tree.getSampleAtAcceptedPosition(), tree.getLogOfMasterPAtAcceptedPosition());
    }

    @Override
    public void step() {

        initializeMomentumForEachVariable(latentVariables, tree.getForwardMomentum(), random);
        cache(tree.getForwardMomentum(), tree.getBackwardMomentum());

        double logOfMasterPMinusMomentumBeforeLeapfrog = tree.getLogOfMasterPAtAcceptedPosition() - 0.5 * dotProduct(tree.getForwardMomentum());

        double logU = Math.log(random.nextDouble()) + logOfMasterPMinusMomentumBeforeLeapfrog;

        int treeHeight = 0;
        tree.resetTreeBeforeSample();

        while (tree.shouldContinue() && treeHeight < maxTreeHeight) {

            //build tree direction -1 = backwards OR 1 = forwards
            int buildDirection = random.nextBoolean() ? 1 : -1;

            Tree otherHalfTree = Tree.buildOtherHalfOfTree(
                tree,
                latentVariables,
                logProbGradientCalculator,
                sampleFromVariables,
                logU,
                buildDirection,
                treeHeight,
                stepsize.getStepsize(),
                logOfMasterPMinusMomentumBeforeLeapfrog,
                random
            );

            if (otherHalfTree.shouldContinue()) {
                final double acceptanceProb = (double) otherHalfTree.getAcceptedLeapfrogCount() / tree.getAcceptedLeapfrogCount();

                Tree.acceptOtherPositionWithProbability(
                    acceptanceProb,
                    tree,
                    otherHalfTree,
                    random
                );
            }

            tree.incrementLeapfrogCount(otherHalfTree.getAcceptedLeapfrogCount());
            tree.setDeltaLikelihoodOfLeapfrog(otherHalfTree.getDeltaLikelihoodOfLeapfrog());
            tree.setTreeSize(otherHalfTree.getTreeSize());
            tree.continueIfNotUTurning(otherHalfTree);

            treeHeight++;
        }

        if (saveStatistics) {
            recordSamplerStatistics();
        }

        if (this.adaptEnabled) {
            stepsize.adaptStepSize(tree, sampleNum);
        }

        tree.acceptPositionAndGradient();
        sampleNum++;
    }

    private void recordSamplerStatistics() {
        stepsize.save(statistics);
        tree.save(statistics);
    }

    private static void initializeMomentumForEachVariable(List<? extends Variable<DoubleTensor>> variables,
                                                          Map<VariableReference, DoubleTensor> momentums,
                                                          KeanuRandom random) {
        for (Variable<DoubleTensor> variable : variables) {
            momentums.put(variable.getReference(), random.nextGaussian(variable.getShape()));
        }
    }

    private static void cache(Map<? extends VariableReference, DoubleTensor> from, Map<VariableReference, DoubleTensor> to) {
        for (Map.Entry<? extends VariableReference, DoubleTensor> entry : from.entrySet()) {
            to.put(entry.getKey(), entry.getValue());
        }
    }

    private static double dotProduct(Map<? extends VariableReference, DoubleTensor> momentums) {
        double dotProduct = 0.0;
        for (DoubleTensor momentum : momentums.values()) {
            dotProduct += momentum.pow(2).sum();
        }
        return dotProduct;
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

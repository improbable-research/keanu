package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.SaveStatistics;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.Getter;
import lombok.Setter;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.SamplingAlgorithm.takeSample;


/**
 * Used by NUTS as a balanced binary tree to compute and store information
 * about leapfrogs that are taken forwards and backwards in space.
 * <p>
 * The tree is reset for each sample.
 */
class Tree implements SaveStatistics {

    private static final double DELTA_MAX = 1000.0;

    @Getter
    private Leapfrog forward;

    @Getter
    private Leapfrog backward;

    @Getter
    @Setter
    private Map<VariableReference, DoubleTensor> sumMomentum;

    @Getter
    @Setter
    private Proposal proposal;

    @Getter
    @Setter
    private double logSumWeight;

    @Getter
    @Setter
    private boolean shouldContinueFlag;

    @Getter
    @Setter
    private double sumMetropolisAcceptanceProbability;

    @Getter
    @Setter
    private int treeSize;

    @Getter
    private double startEnergy;

    /**
     * @param startState                         The leap frog for is initial step in the tree.
     * @param proposal                           The current accepted position.
     * @param logSumWeight                       ?????
     * @param shouldContinueFlag                 False if the tree is U-Turning
     * @param sumMetropolisAcceptanceProbability A summation of the metropolis acceptance probability
     *                                           over each step in tree.
     * @param treeSize                           The size of the tree.
     * @param startEnergy                        The
     */
    Tree(Leapfrog startState,
         Proposal proposal,
         double logSumWeight,
         boolean shouldContinueFlag,
         double sumMetropolisAcceptanceProbability,
         int treeSize,
         double startEnergy) {

        this.forward = startState;
        this.backward = startState;
        this.sumMomentum = startState.getMomentum();
        this.proposal = proposal;
        this.logSumWeight = logSumWeight;
        this.shouldContinueFlag = shouldContinueFlag;
        this.sumMetropolisAcceptanceProbability = sumMetropolisAcceptanceProbability;
        this.treeSize = treeSize;
        this.startEnergy = startEnergy;
    }


    public static Tree buildOtherHalfOfTree(Tree currentTree,
                                            List<? extends Variable<DoubleTensor, ?>> latentVariables,
                                            ProbabilisticModelWithGradient logProbGradientCalculator,
                                            final List<? extends Variable> sampleFromVariables,
                                            int buildDirection,
                                            int treeHeight,
                                            double epsilon,
                                            KeanuRandom random) {

        Tree otherHalfTree = buildTree(
            latentVariables,
            logProbGradientCalculator,
            sampleFromVariables,
            buildDirection == -1 ? currentTree.backward : currentTree.forward,
            buildDirection,
            treeHeight,
            epsilon,
            currentTree.startEnergy,
            random
        );

        if (buildDirection == -1) {
            currentTree.backward = otherHalfTree.backward;
        } else {
            currentTree.forward = otherHalfTree.forward;
        }

        return otherHalfTree;
    }

    private static Tree buildTree(List<? extends Variable<DoubleTensor, ?>> latentVariables,
                                  ProbabilisticModelWithGradient logProbGradientCalculator,
                                  final List<? extends Variable> sampleFromVariables,
                                  Leapfrog leapfrog,
                                  int buildDirection,
                                  int treeHeight,
                                  double epsilon,
                                  double startEnergy,
                                  KeanuRandom random) {
        if (treeHeight == 0) {

            //Base case-take one leapfrog step in the build direction

            return treeBuilderBaseCase(
                latentVariables,
                logProbGradientCalculator,
                sampleFromVariables,
                leapfrog,
                buildDirection,
                epsilon,
                startEnergy
            );

        } else {
            //Recursion-implicitly build the left and right subtrees.

            Tree tree = buildTree(
                latentVariables,
                logProbGradientCalculator,
                sampleFromVariables,
                leapfrog,
                buildDirection,
                treeHeight - 1,
                epsilon,
                startEnergy,
                random
            );

            //Should continue building other half if first half's shouldContinueFlag is true
            if (tree.shouldContinueFlag) {

                Tree otherHalfTree = buildOtherHalfOfTree(
                    tree,
                    latentVariables,
                    logProbGradientCalculator,
                    sampleFromVariables,
                    buildDirection,
                    treeHeight - 1,
                    epsilon,
                    random
                );

                if (otherHalfTree.shouldContinueFlag) {

                    tree.setSumMomentum(VariableValues.add(tree.sumMomentum, otherHalfTree.sumMomentum));

                    boolean notUTurning = isNotUTurning(tree.forward.getVelocity(), tree.backward.getVelocity(), tree.sumMomentum);

                    tree.setShouldContinueFlag(notUTurning);

                    final double newLogSize = logSumExp(tree.logSumWeight, otherHalfTree.logSumWeight);

                    tree.setLogSumWeight(newLogSize);

                    if (acceptOtherPositionWithProbability(otherHalfTree.logSumWeight - newLogSize, random)) {
                        tree.setProposal(otherHalfTree.proposal);
                    }

                }

                tree.sumMetropolisAcceptanceProbability += otherHalfTree.sumMetropolisAcceptanceProbability;
                tree.treeSize += otherHalfTree.treeSize;
            }

            return tree;
        }

    }

    public static double logSumExp(double a, double b) {
        double max = Math.max(a, b);
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    }

    private static Tree treeBuilderBaseCase(final List<? extends Variable<DoubleTensor, ?>> latentVariables,
                                            final ProbabilisticModelWithGradient logProbGradientCalculator,
                                            final List<? extends Variable> sampleFromVariables,
                                            final Leapfrog leapfrog,
                                            final int buildDirection,
                                            final double epsilon,
                                            final double startEnergy) {

        Leapfrog leapfrogAfterStep = leapfrog.step(latentVariables, logProbGradientCalculator, epsilon * buildDirection);

        final double energyAfterStep = leapfrogAfterStep.getEnergy();

        final double energyChange = energyAfterStep - startEnergy;

        final boolean shouldContinueFlag = Math.abs(energyChange) < DELTA_MAX;

        if (shouldContinueFlag) {

            final double logSumWeight = -energyChange;

            final double metropolisAcceptanceProbability = Math.min(
                1.0,
                Math.exp(logSumWeight)
            );

            final Map<VariableReference, ?> sample = takeSample(sampleFromVariables);

            if (isNotUsableNumber(logSumWeight) || isNotUsableNumber(metropolisAcceptanceProbability)) {
                throw new IllegalStateException("acceptProbability is " + metropolisAcceptanceProbability + " logSumWeight is " + logSumWeight);
            }

            Proposal proposal = new Proposal(
                leapfrogAfterStep.getPosition(),
                leapfrogAfterStep.getGradient(),
                sample,
                energyAfterStep,
                metropolisAcceptanceProbability,
                leapfrogAfterStep.getLogProb()
            );

            return new Tree(
                leapfrogAfterStep,
                proposal,
                logSumWeight,
                true,
                metropolisAcceptanceProbability,
                1,
                startEnergy
            );
        } else {

            return new Tree(
                leapfrogAfterStep,
                null,
                Double.NEGATIVE_INFINITY,
                false,
                0,
                1,
                startEnergy
            );
        }
    }

    public static boolean isNotUsableNumber(double value) {
        return Double.isInfinite(value) || Double.isNaN(value);
    }

    public static boolean acceptOtherPositionWithProbability(double probability,
                                                             KeanuRandom random) {

        if (isNotUsableNumber(probability)) {
            throw new IllegalStateException("Accept probability is " + probability);
        }

        return Math.log(random.nextDouble()) < probability;
    }

    public static boolean isNotUTurning(Map<VariableReference, DoubleTensor> velocityForward,
                                        Map<VariableReference, DoubleTensor> velocityBackward,
                                        Map<VariableReference, DoubleTensor> rho) {
        double forward = 0.0;
        double backward = 0.0;

        for (VariableReference latentId : velocityForward.keySet()) {

            final DoubleTensor vForward = velocityForward.get(latentId);
            final DoubleTensor vBackward = velocityBackward.get(latentId);
            final DoubleTensor rhoForLatent = rho.get(latentId);

            forward += vForward.times(rhoForLatent).sum();
            backward += vBackward.times(rhoForLatent).sum();
        }

        if (isNotUsableNumber(forward) || isNotUsableNumber(backward)) {
            throw new IllegalStateException("Forward : " + forward + " Backward : " + backward);
        }

        return (forward >= 0.0) && (backward >= 0.0);
    }

    public boolean shouldContinue() {
        return shouldContinueFlag;
    }

    public void save(Statistics statistics) {
        statistics.store(NUTS.Metrics.LOG_PROB, proposal.getLogProb());
        statistics.store(NUTS.Metrics.TREE_SIZE, (double) treeSize);
    }
}
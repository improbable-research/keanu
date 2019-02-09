package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.SaveStatistics;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.SamplingAlgorithm.takeSample;
import static io.improbable.keanu.algorithms.mcmc.nuts.VariableValues.add;


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
    private final double startEnergy;

    private final ProbabilisticModelWithGradient logProbGradientCalculator;

    private final List<? extends Variable> sampleFromVariables;

    private final KeanuRandom random;

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
         double startEnergy,
         ProbabilisticModelWithGradient logProbGradientCalculator,
         List<? extends Variable> sampleFromVariables,
         KeanuRandom random) {

        this.forward = startState;
        this.backward = startState;
        this.sumMomentum = startState.getMomentum();
        this.proposal = proposal;
        this.logSumWeight = logSumWeight;
        this.shouldContinueFlag = shouldContinueFlag;
        this.sumMetropolisAcceptanceProbability = sumMetropolisAcceptanceProbability;
        this.treeSize = treeSize;
        this.startEnergy = startEnergy;
        this.logProbGradientCalculator = logProbGradientCalculator;
        this.sampleFromVariables = sampleFromVariables;
        this.random = random;
    }

    public void buildTree(int treeHeight,
                          int buildDirection,
                          double epsilon) {

        SubTree otherHalfTree = buildTree(
            buildDirection == -1 ? backward : forward,
            buildDirection,
            treeHeight,
            epsilon
        );

        if (buildDirection == -1) {
            backward = otherHalfTree.backward;
        } else {
            forward = otherHalfTree.forward;
        }

        sumMetropolisAcceptanceProbability += otherHalfTree.sumMetropolisAcceptanceProbability;
        treeSize += otherHalfTree.treeSize;

        if (otherHalfTree.shouldContinueFlag) {


            if (Tree.acceptOtherProposalWithProbability(otherHalfTree.getLogSumWeight() - logSumWeight, random)) {
                proposal = otherHalfTree.proposal;
            }

            logSumWeight = logSumExp(logSumWeight, otherHalfTree.logSumWeight);

            sumMomentum = add(sumMomentum, otherHalfTree.sumMomentum);
        }

        shouldContinueFlag = otherHalfTree.shouldContinueFlag && Tree.isNotUTurning(
            forward.getVelocity(),
            backward.getVelocity(),
            sumMomentum
        );
    }

    private SubTree buildTree(Leapfrog leapfrog,
                              int buildDirection,
                              int treeHeight,
                              double epsilon) {
        if (treeHeight == 0) {

            //Base case-take one leapfrog step in the build direction

            return treeBuilderBaseCase(
                leapfrog,
                buildDirection,
                epsilon
            );

        } else {
            //Recursion-implicitly build the left and right subtrees.

            SubTree subTree = buildTree(
                leapfrog,
                buildDirection,
                treeHeight - 1,
                epsilon
            );

            //Should continue building other half if first half's shouldContinueFlag is true
            if (subTree.shouldContinueFlag) {

                SubTree rightSubtree = buildTree(
                    buildDirection == -1 ? subTree.backward : subTree.forward,
                    buildDirection,
                    treeHeight - 1,
                    epsilon
                );

                if (buildDirection == -1) {
                    subTree.backward = rightSubtree.backward;
                } else {
                    subTree.forward = rightSubtree.forward;
                }

                if (rightSubtree.shouldContinueFlag) {

                    subTree.sumMomentum = add(subTree.sumMomentum, rightSubtree.sumMomentum);

                    subTree.shouldContinueFlag = isNotUTurning(
                        subTree.forward.getVelocity(),
                        subTree.backward.getVelocity(),
                        subTree.sumMomentum
                    );

                    final double newLogSize = logSumExp(subTree.logSumWeight, rightSubtree.logSumWeight);

                    subTree.logSumWeight = newLogSize;

                    if (acceptOtherProposalWithProbability(rightSubtree.logSumWeight - newLogSize, random)) {
                        subTree.setProposal(rightSubtree.proposal);
                    }

                }

                subTree.sumMetropolisAcceptanceProbability += rightSubtree.sumMetropolisAcceptanceProbability;
                subTree.treeSize += rightSubtree.treeSize;
            }

            return subTree;
        }

    }

    public static double logSumExp(double a, double b) {
        double max = Math.max(a, b);
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    }

    private SubTree treeBuilderBaseCase(final Leapfrog leapfrog,
                                        final int buildDirection,
                                        final double epsilon) {

        Leapfrog leapfrogAfterStep = leapfrog.step(logProbGradientCalculator, epsilon * buildDirection);

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
                leapfrogAfterStep.getLogProb()
            );

            return new SubTree(
                leapfrogAfterStep,
                leapfrogAfterStep,
                leapfrogAfterStep.getMomentum(),
                proposal,
                logSumWeight,
                true,
                metropolisAcceptanceProbability,
                1
            );

        } else {

            return new SubTree(
                leapfrogAfterStep,
                leapfrogAfterStep,
                leapfrogAfterStep.getMomentum(),
                null,
                Double.NEGATIVE_INFINITY,
                false,
                0,
                1
            );
        }
    }

    public static boolean isNotUsableNumber(double value) {
        return Double.isInfinite(value) || Double.isNaN(value);
    }

    private static boolean acceptOtherProposalWithProbability(double probability,
                                                              KeanuRandom random) {

        if (isNotUsableNumber(probability)) {
            throw new IllegalStateException("Accept probability is " + probability);
        }

        return Math.log(random.nextDouble()) < probability;
    }

    private static boolean isNotUTurning(Map<VariableReference, DoubleTensor> velocityForward,
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

    @Data
    @AllArgsConstructor
    private static class SubTree {

        /**
         * The leap frog at the forward most step in the tree.
         */
        private Leapfrog forward;

        /**
         * The leap frog at the backward most step in the tree
         */
        private Leapfrog backward;


        /**
         * The sum of all of the momentum from each step
         */
        private Map<VariableReference, DoubleTensor> sumMomentum;

        /**
         * The current accepted proposal.
         */
        private Proposal proposal;

        /**
         * ????
         */
        private double logSumWeight;

        /**
         * A flag indicating either the steps are turning or have diverged due to significant energy change.
         */
        private boolean shouldContinueFlag;

        /**
         * A summation of the metropolis acceptance probability from each step
         */
        private double sumMetropolisAcceptanceProbability;

        /**
         * The size of the sub tree, which is the number of steps taken to build the tree.
         */
        private int treeSize;
    }
}
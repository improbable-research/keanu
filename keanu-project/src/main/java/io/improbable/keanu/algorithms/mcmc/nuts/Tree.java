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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
    private Leapfrog leapfrogForward;

    @Getter
    private Leapfrog leapfrogBackward;

    @Getter
    @Setter
    private Map<VariableReference, DoubleTensor> pSum;

    @Getter
    @Setter
    private Proposal proposal;

    @Getter
    @Setter
    private double logSize;

    @Getter
    @Setter
    private boolean shouldContinueFlag;

    @Getter
    @Setter
    private double acceptSum;

    @Getter
    @Setter
    private int treeSize;

    @Getter
    private double startEnergy;

    /**
     * @param startState         The result of the forward leapfrog
     * @param proposal           ??
     * @param logSize            ?????
     * @param shouldContinueFlag False if we are U-Turning
     * @param acceptSum          ?????
     * @param treeSize           The size of the tree
     */
    Tree(Leapfrog startState,
         Proposal proposal,
         double logSize,
         boolean shouldContinueFlag,
         double acceptSum,
         int treeSize,
         double startEnergy) {

        this.leapfrogForward = startState;
        this.leapfrogBackward = startState;
        this.pSum = startState.getMomentum();
        this.proposal = proposal;
        this.logSize = logSize;
        this.shouldContinueFlag = shouldContinueFlag;
        this.acceptSum = acceptSum;
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
            buildDirection == -1 ? currentTree.leapfrogBackward : currentTree.leapfrogForward,
            buildDirection,
            treeHeight,
            epsilon,
            currentTree.startEnergy,
            random
        );

        if (buildDirection == -1) {
            currentTree.leapfrogBackward = otherHalfTree.leapfrogBackward;
        } else {
            currentTree.leapfrogForward = otherHalfTree.leapfrogForward;
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

                    tree.setPSum(add(tree.pSum, otherHalfTree.pSum));

                    boolean notUTurning = isNotUTurning(tree.leapfrogForward.getVelocity(), tree.leapfrogBackward.getVelocity(), tree.pSum);

                    tree.setShouldContinueFlag(notUTurning);

                    final double newLogSize = logSumExp(tree.logSize, otherHalfTree.logSize);

                    if (Tree.isNotUsableNumber(newLogSize)) {
                        throw new IllegalStateException("New logsize is " + newLogSize);
                    }

                    tree.setLogSize(newLogSize);

                    if (acceptOtherPositionWithProbability(otherHalfTree.logSize - newLogSize, random)) {
                        tree.setProposal(otherHalfTree.proposal);
                    }

                }

                tree.acceptSum += otherHalfTree.acceptSum;
                tree.treeSize += otherHalfTree.treeSize;
            }

            return tree;
        }

    }

    public static double logSumExp(double a, double b) {
        double max = Math.max(a, b);
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    }

    public static Map<VariableReference, DoubleTensor> add(Map<VariableReference, DoubleTensor> a, Map<VariableReference, DoubleTensor> b) {
        Map<VariableReference, DoubleTensor> sum = new HashMap<>();
        for (Map.Entry<VariableReference, DoubleTensor> e : a.entrySet()) {
            sum.put(e.getKey(), e.getValue().plus(b.get(e.getKey())));
        }
        return sum;
    }

    private static Set<Leapfrog> leapfrogs = new HashSet<>();

    private static Tree treeBuilderBaseCase(final List<? extends Variable<DoubleTensor, ?>> latentVariables,
                                            final ProbabilisticModelWithGradient logProbGradientCalculator,
                                            final List<? extends Variable> sampleFromVariables,
                                            final Leapfrog leapfrog,
                                            final int buildDirection,
                                            final double epsilon,
                                            final double startEnergy) {
        leapfrogs.add(leapfrog);

        Leapfrog leapfrogAfterStep = leapfrog.step(latentVariables, logProbGradientCalculator, epsilon * buildDirection);

        leapfrogs.add(leapfrogAfterStep);

        final double energyAfterStep = leapfrogAfterStep.getEnergy();

        final double energyChange = energyAfterStep - startEnergy;

        final boolean shouldContinueFlag = Math.abs(energyChange) < DELTA_MAX;

        if (shouldContinueFlag) {

            final double logSize = -energyChange;

            final double pAccept = Math.min(
                1.0,
                Math.exp(logSize)
            );

            final Map<VariableReference, ?> sample = takeSample(sampleFromVariables);

            if (isNotUsableNumber(logSize) || isNotUsableNumber(pAccept)) {
                throw new IllegalStateException("acceptSum is " + pAccept + " logSize is " + logSize);
            }

            Proposal proposal = new Proposal(
                leapfrogAfterStep.getPosition(),
                leapfrogAfterStep.getGradient(),
                sample,
                energyAfterStep,
                pAccept,
                leapfrogAfterStep.getLogProb()
            );

            return new Tree(
                leapfrogAfterStep,
                proposal,
                logSize,
                true,
                pAccept,
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
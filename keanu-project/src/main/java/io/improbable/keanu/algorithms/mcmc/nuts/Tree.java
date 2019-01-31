package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.SaveStatistics;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

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
    private static final int STARTING_ACCEPTED_LEAPFROG = 1;
    private static final boolean STARTING_SHOULD_CONTINUE = true;
    private static final double STARTING_DELTA_LIKELIHOOD = 0.;
    private static final int STARTING_TREE_SIZE = 1;

    private Leapfrog leapfrogForward;
    private Leapfrog leapfrogBackward;
    private Map<VariableReference, DoubleTensor> acceptedPosition;
    private Map<? extends VariableReference, DoubleTensor> gradientAtAcceptedPosition;
    private double logOfMasterPAtAcceptedPosition;
    private Map<VariableReference, ?> sampleAtAcceptedPosition;
    private int acceptedLeapfrogCount;
    private boolean shouldContinueFlag;
    private double deltaLikelihoodOfLeapfrog;
    private double treeSize;

    /**
     * @param leapfrog                  The result of the forward leapfrog
     * @param logProbAtAcceptedPosition The log prob of the network at the accepted position
     * @param sampleAtAcceptedPosition  The sample value at the accepted position
     * @param acceptedLeapfrogCount     The number of accepted leapfrogs
     * @param shouldContinueFlag        False if we are U-Turning
     * @param deltaLikelihoodOfLeapfrog The change in log prob as a result of the latest leapfrog
     * @param treeSize                  The size of the tree
     */
    Tree(Leapfrog leapfrog,
         double logProbAtAcceptedPosition,
         Map<VariableReference, ?> sampleAtAcceptedPosition,
         int acceptedLeapfrogCount,
         boolean shouldContinueFlag,
         double deltaLikelihoodOfLeapfrog,
         double treeSize) {

        this.leapfrogForward = leapfrog;
        this.leapfrogBackward = leapfrog;
        this.acceptedPosition = leapfrog.getPosition();
        this.gradientAtAcceptedPosition = leapfrog.getGradient();
        this.logOfMasterPAtAcceptedPosition = logProbAtAcceptedPosition;
        this.sampleAtAcceptedPosition = sampleAtAcceptedPosition;
        this.acceptedLeapfrogCount = acceptedLeapfrogCount;
        this.shouldContinueFlag = shouldContinueFlag;
        this.deltaLikelihoodOfLeapfrog = deltaLikelihoodOfLeapfrog;
        this.treeSize = treeSize;
    }

    /**
     * A helper method to create a basic tree
     *
     * @param position                 the starting position
     * @param momentum                 the starting momentum
     * @param gradient                 the starting gradient
     * @param initialLogOfMasterP      the initial log prob
     * @param sampleAtAcceptedPosition the initial sample
     * @return a basic tree
     */
    public static Tree createInitialTree(Map<VariableReference, DoubleTensor> position,
                                         Map<VariableReference, DoubleTensor> momentum,
                                         Map<? extends
                                             VariableReference, DoubleTensor> gradient,
                                         double initialLogOfMasterP,
                                         Map<VariableReference, ?> sampleAtAcceptedPosition) {

        return new Tree(
            new Leapfrog(position, momentum, gradient),
            initialLogOfMasterP,
            sampleAtAcceptedPosition,
            STARTING_ACCEPTED_LEAPFROG,
            STARTING_SHOULD_CONTINUE,
            STARTING_DELTA_LIKELIHOOD,
            STARTING_TREE_SIZE
        );
    }

    public static Tree buildOtherHalfOfTree(Tree currentTree,
                                            List<? extends Variable<DoubleTensor, ?>> latentVariables,
                                            ProbabilisticModelWithGradient logProbGradientCalculator,
                                            final List<? extends Variable> sampleFromVariables,
                                            double logU,
                                            int buildDirection,
                                            int treeHeight,
                                            double epsilon,
                                            double logOfMasterPMinusMomentumBeforeLeapfrog,
                                            KeanuRandom random) {

        Tree otherHalfTree = buildTree(
            latentVariables,
            logProbGradientCalculator,
            sampleFromVariables,
            buildDirection == -1 ? currentTree.leapfrogBackward : currentTree.leapfrogForward,
            logU,
            buildDirection,
            treeHeight,
            epsilon,
            logOfMasterPMinusMomentumBeforeLeapfrog,
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
                                  double logU,
                                  int buildDirection,
                                  int treeHeight,
                                  double epsilon,
                                  double logOfMasterPMinusMomentumBeforeLeapfrog,
                                  KeanuRandom random) {
        if (treeHeight == 0) {

            //Base case-take one leapfrog step in the build direction

            return treeBuilderBaseCase(
                latentVariables,
                logProbGradientCalculator,
                sampleFromVariables,
                leapfrog,
                logU,
                buildDirection,
                epsilon,
                logOfMasterPMinusMomentumBeforeLeapfrog
            );

        } else {
            //Recursion-implicitly build the left and right subtrees.

            Tree tree = buildTree(
                latentVariables,
                logProbGradientCalculator,
                sampleFromVariables,
                leapfrog,
                logU,
                buildDirection,
                treeHeight - 1,
                epsilon,
                logOfMasterPMinusMomentumBeforeLeapfrog,
                random
            );

            //Should continue building other half if first half's shouldContinueFlag is true
            if (tree.shouldContinueFlag) {

                Tree otherHalfTree = buildOtherHalfOfTree(
                    tree,
                    latentVariables,
                    logProbGradientCalculator,
                    sampleFromVariables,
                    logU,
                    buildDirection,
                    treeHeight - 1,
                    epsilon,
                    logOfMasterPMinusMomentumBeforeLeapfrog,
                    random
                );

                double acceptOtherTreePositionProbability = (double) otherHalfTree.acceptedLeapfrogCount / (tree.acceptedLeapfrogCount + otherHalfTree.acceptedLeapfrogCount);

                Tree.acceptOtherPositionWithProbability(
                    acceptOtherTreePositionProbability,
                    tree,
                    otherHalfTree,
                    random
                );

                tree.continueIfNotUTurning(otherHalfTree);

                tree.acceptedLeapfrogCount += otherHalfTree.acceptedLeapfrogCount;
                tree.deltaLikelihoodOfLeapfrog += otherHalfTree.deltaLikelihoodOfLeapfrog;
                tree.treeSize += otherHalfTree.treeSize;
            }

            return tree;
        }

    }

    private static Tree treeBuilderBaseCase(List<? extends Variable<DoubleTensor, ?>> latentVariables,
                                            ProbabilisticModelWithGradient logProbGradientCalculator,
                                            final List<? extends Variable> sampleFromVariables,
                                            Leapfrog leapfrog,
                                            double logU,
                                            int buildDirection,
                                            double epsilon,
                                            double logOfMasterPMinusMomentumBeforeLeapfrog) {

        leapfrog = leapfrog.step(latentVariables, logProbGradientCalculator, epsilon * buildDirection);

        final double logOfMasterPAfterLeapfrog = logProbGradientCalculator.logProb();

        final double logOfMasterPMinusMomentum = logOfMasterPAfterLeapfrog - leapfrog.halfDotProductMomentum();
        final int acceptedLeapfrogCount = logU <= logOfMasterPMinusMomentum ? 1 : 0;
        final boolean shouldContinueFlag = logU < DELTA_MAX + logOfMasterPMinusMomentum;

        final Map<VariableReference, ?> sampleAtAcceptedPosition = takeSample((List<? extends Variable<Object, ?>>)sampleFromVariables);

        final double deltaLikelihoodOfLeapfrog = Math.min(
            1.0,
            Math.exp(logOfMasterPMinusMomentum - logOfMasterPMinusMomentumBeforeLeapfrog)
        );

        return new Tree(
            leapfrog,
            logOfMasterPAfterLeapfrog,
            sampleAtAcceptedPosition,
            acceptedLeapfrogCount,
            shouldContinueFlag,
            deltaLikelihoodOfLeapfrog,
            STARTING_TREE_SIZE
        );
    }

    public static void acceptOtherPositionWithProbability(double probability,
                                                          Tree tree,
                                                          Tree otherTree,
                                                          KeanuRandom random) {
        if (random.nextDouble() < probability) {
            tree.acceptedPosition = otherTree.acceptedPosition;
            tree.gradientAtAcceptedPosition = otherTree.gradientAtAcceptedPosition;
            tree.logOfMasterPAtAcceptedPosition = otherTree.logOfMasterPAtAcceptedPosition;
            tree.sampleAtAcceptedPosition = otherTree.sampleAtAcceptedPosition;
        }
    }

    private static boolean isNotUTurning(Map<VariableReference, DoubleTensor> positionForward,
                                         Map<VariableReference, DoubleTensor> positionBackward,
                                         Map<VariableReference, DoubleTensor> momentumForward,
                                         Map<VariableReference, DoubleTensor> momentumBackward) {
        double forward = 0.0;
        double backward = 0.0;

        for (Map.Entry<VariableReference, DoubleTensor> forwardPositionForLatent : positionForward.entrySet()) {

            final VariableReference latentId = forwardPositionForLatent.getKey();
            final DoubleTensor forwardMinusBackward = forwardPositionForLatent.getValue().minus(
                positionBackward.get(latentId)
            );

            forward += forwardMinusBackward.times(momentumForward.get(latentId)).sum();
            backward += forwardMinusBackward.timesInPlace(momentumBackward.get(latentId)).sum();
        }

        return (forward >= 0.0) && (backward >= 0.0);
    }

    public void continueIfNotUTurning(Tree otherHalfTree) {
        shouldContinueFlag = (otherHalfTree.shouldContinue() && Tree.isNotUTurning(
            getForwardPosition(),
            getBackwardPosition(),
            getForwardMomentum(),
            getBackwardMomentum()
        ));
    }

    public boolean shouldContinue() {
        return shouldContinueFlag;
    }

    public int getAcceptedLeapfrogCount() {
        return acceptedLeapfrogCount;
    }

    public void resetTreeBeforeSample() {
        this.shouldContinueFlag = true;
        this.acceptedLeapfrogCount = 1;
    }

    public double getDeltaLikelihoodOfLeapfrog() {
        return deltaLikelihoodOfLeapfrog;
    }

    public void setDeltaLikelihoodOfLeapfrog(double deltaLikelihoodOfLeapfrog) {
        this.deltaLikelihoodOfLeapfrog = deltaLikelihoodOfLeapfrog;
    }

    public double getTreeSize() {
        return treeSize;
    }

    public void setTreeSize(double treeSize) {
        this.treeSize = treeSize;
    }

    public void incrementLeapfrogCount(int otherTreeAcceptedCount) {
        this.acceptedLeapfrogCount += otherTreeAcceptedCount;
    }

    public Map<VariableReference, ?> getSampleAtAcceptedPosition() {
        return sampleAtAcceptedPosition;
    }

    public double getLogOfMasterPAtAcceptedPosition() {
        return logOfMasterPAtAcceptedPosition;
    }

    public Map<VariableReference, DoubleTensor> getForwardPosition() {
        return leapfrogForward.getPosition();
    }

    public Map<VariableReference, DoubleTensor> getBackwardPosition() {
        return leapfrogBackward.getPosition();
    }

    public Map<VariableReference, DoubleTensor> getForwardMomentum() {
        return leapfrogForward.getMomentum();
    }

    public Map<VariableReference, DoubleTensor> getBackwardMomentum() {
        return leapfrogBackward.getMomentum();
    }

    public void acceptPositionAndGradient() {
        leapfrogForward = leapfrogForward.makeJumpTo(acceptedPosition, gradientAtAcceptedPosition);
        leapfrogBackward = leapfrogBackward.makeJumpTo(acceptedPosition, gradientAtAcceptedPosition);
    }

    public void save(Statistics statistics) {
        statistics.store(NUTS.Metrics.LOG_PROB, logOfMasterPAtAcceptedPosition);
        statistics.store(NUTS.Metrics.TREE_SIZE, treeSize);
    }
}
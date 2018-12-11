package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Used by NUTS as a balanced binary tree to compute and store information
 * about leapfrogs that are taken forwards and backwards in space.
 *
 * The tree is reset for each sample.
 */
public class TreeBuilder {

    private static final double DELTA_MAX = 1000.0;
    private static final int STARTING_ACCEPTED_LEAPFROG = 1;
    private static final boolean STARTING_SHOULD_CONTINUE = true;
    private static final double STARTING_DELTA_LIKELIHOOD = 0.;
    private static final int STARTING_TREE_SIZE = 1;

    private Leapfrog leapfrogForward;
    private Leapfrog leapfrogBackward;
    private Map<VertexId, DoubleTensor> acceptedPosition;
    private Map<VertexId, DoubleTensor> gradientAtAcceptedPosition;
    private double logOfMasterPAtAcceptedPosition;
    private Map<VertexId, ?> sampleAtAcceptedPosition;
    private int acceptedLeapfrogCount;
    private boolean shouldContinueFlag;
    private double deltaLikelihoodOfLeapfrog;
    private double treeSize;

    /**
     * @param leapfrogForward            The result of the forward leapfrog
     * @param leapfrogBackward           The result of the backward leapfrog
     * @param acceptedPosition           The accepted position of the vertices
     * @param gradientAtAcceptedPosition The gradient at the accepted position
     * @param logProbAtAcceptedPosition  The log prob of the network at the accepted position
     * @param sampleAtAcceptedPosition   The sample value at the accepted position
     * @param acceptedLeapfrogCount      The number of accepted leapfrogs
     * @param shouldContinueFlag         False if we are U-Turning
     * @param deltaLikelihoodOfLeapfrog  The change in log prob as a result of the latest leapfrog
     * @param treeSize                   The size of the tree
     */
    TreeBuilder(Leapfrog leapfrogForward,
                Leapfrog leapfrogBackward,
                Map<VertexId, DoubleTensor> acceptedPosition,
                Map<VertexId, DoubleTensor> gradientAtAcceptedPosition,
                double logProbAtAcceptedPosition,
                Map<VertexId, ?> sampleAtAcceptedPosition,
                int acceptedLeapfrogCount,
                boolean shouldContinueFlag,
                double deltaLikelihoodOfLeapfrog,
                double treeSize) {

        this.leapfrogForward = leapfrogForward;
        this.leapfrogBackward = leapfrogBackward;
        this.acceptedPosition = acceptedPosition;
        this.gradientAtAcceptedPosition = gradientAtAcceptedPosition;
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
    public static TreeBuilder createBasicTree(Map<VertexId, DoubleTensor> position,
                                              Map<VertexId, DoubleTensor> momentum,
                                              Map<VertexId, DoubleTensor> gradient,
                                              double initialLogOfMasterP,
                                              Map<VertexId, ?> sampleAtAcceptedPosition) {

        return new TreeBuilder(
            new Leapfrog(position, momentum, gradient),
            new Leapfrog(position, momentum, gradient),
            position,
            gradient,
            initialLogOfMasterP,
            sampleAtAcceptedPosition,
            STARTING_ACCEPTED_LEAPFROG,
            STARTING_SHOULD_CONTINUE,
            STARTING_DELTA_LIKELIHOOD,
            STARTING_TREE_SIZE
        );
    }

    public static TreeBuilder buildOtherHalfOfTree(TreeBuilder currentTree,
                                                   List<Vertex<DoubleTensor>> latentVertices,
                                                   List<Vertex> probabilisticVertices,
                                                   LogProbGradientCalculator logProbGradientCalculator,
                                                   final List<? extends Vertex> sampleFromVertices,
                                                   double logU,
                                                   int buildDirection,
                                                   int treeHeight,
                                                   double epsilon,
                                                   double logOfMasterPMinusMomentumBeforeLeapfrog,
                                                   KeanuRandom random) {

        TreeBuilder otherHalfTree;

        if (buildDirection == -1) {

            otherHalfTree = buildTree(
                latentVertices,
                probabilisticVertices,
                logProbGradientCalculator,
                sampleFromVertices,
                currentTree.leapfrogBackward,
                logU,
                buildDirection,
                treeHeight,
                epsilon,
                logOfMasterPMinusMomentumBeforeLeapfrog,
                random
            );

            currentTree.leapfrogBackward = otherHalfTree.leapfrogBackward;

        } else {

            otherHalfTree = buildTree(
                latentVertices,
                probabilisticVertices,
                logProbGradientCalculator,
                sampleFromVertices,
                currentTree.leapfrogForward,
                logU,
                buildDirection,
                treeHeight,
                epsilon,
                logOfMasterPMinusMomentumBeforeLeapfrog,
                random
            );

            currentTree.leapfrogForward = otherHalfTree.leapfrogForward;
        }

        return otherHalfTree;
    }

    public static TreeBuilder buildTree(List<Vertex<DoubleTensor>> latentVertices,
                                        List<Vertex> probabilisticVertices,
                                        LogProbGradientCalculator logProbGradientCalculator,
                                        final List<? extends Vertex> sampleFromVertices,
                                        Leapfrog leapfrog,
                                        double logU,
                                        int buildDirection,
                                        int treeHeight,
                                        double epsilon,
                                        double logOfMasterPMinusMomentumBeforeLeapfrog,
                                        KeanuRandom random) {
        if (treeHeight == 0) {

            //Base case-take one leapfrog step in the build direction

            return treeBuilderBaseCase(latentVertices,
                probabilisticVertices,
                logProbGradientCalculator,
                sampleFromVertices,
                leapfrog,
                logU,
                buildDirection,
                epsilon,
                logOfMasterPMinusMomentumBeforeLeapfrog
            );

        } else {
            //Recursion-implicitly build the left and right subtrees.

            TreeBuilder tree = buildTree(
                latentVertices,
                probabilisticVertices,
                logProbGradientCalculator,
                sampleFromVertices,
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

                TreeBuilder otherHalfTree = buildOtherHalfOfTree(
                    tree,
                    latentVertices,
                    probabilisticVertices,
                    logProbGradientCalculator,
                    sampleFromVertices,
                    logU,
                    buildDirection,
                    treeHeight - 1,
                    epsilon,
                    logOfMasterPMinusMomentumBeforeLeapfrog,
                    random
                );

                double acceptOtherTreePositionProbability = (double) otherHalfTree.acceptedLeapfrogCount / (tree.acceptedLeapfrogCount + otherHalfTree.acceptedLeapfrogCount);

                TreeBuilder.acceptOtherPositionWithProbability(
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

    public static TreeBuilder treeBuilderBaseCase(List<Vertex<DoubleTensor>> latentVertices,
                                                  List<Vertex> probabilisticVertices,
                                                  LogProbGradientCalculator logProbGradientCalculator,
                                                  final List<? extends Vertex> sampleFromVertices,
                                                  Leapfrog leapfrog,
                                                  double logU,
                                                  int buildDirection,
                                                  double epsilon,
                                                  double logOfMasterPMinusMomentumBeforeLeapfrog) {

        leapfrog = leapfrog.step(latentVertices, logProbGradientCalculator, epsilon * buildDirection);

        final double logOfMasterPAfterLeapfrog = ProbabilityCalculator.calculateLogProbFor(probabilisticVertices);

        final double logOfMasterPMinusMomentum = logOfMasterPAfterLeapfrog - leapfrog.halfDotProductMomentum();
        final int acceptedLeapfrogCount = logU <= logOfMasterPMinusMomentum ? 1 : 0;
        final boolean shouldContinueFlag = logU < DELTA_MAX + logOfMasterPMinusMomentum;

        final Map<VertexId, ?> sampleAtAcceptedPosition = takeSample(sampleFromVertices);

        final double deltaLikelihoodOfLeapfrog = Math.min(
            1.0,
            Math.exp(logOfMasterPMinusMomentum - logOfMasterPMinusMomentumBeforeLeapfrog)
        );

        return new TreeBuilder(
            leapfrog,
            leapfrog,
            leapfrog.getPosition(),
            leapfrog.getGradient(),
            logOfMasterPAfterLeapfrog,
            sampleAtAcceptedPosition,
            acceptedLeapfrogCount,
            shouldContinueFlag,
            deltaLikelihoodOfLeapfrog,
            STARTING_TREE_SIZE
        );
    }

    public static void acceptOtherPositionWithProbability(double probability,
                                                          TreeBuilder tree,
                                                          TreeBuilder otherTree,
                                                          KeanuRandom random) {
        if (withProbability(probability, random)) {
            tree.acceptedPosition = otherTree.acceptedPosition;
            tree.gradientAtAcceptedPosition = otherTree.gradientAtAcceptedPosition;
            tree.logOfMasterPAtAcceptedPosition = otherTree.logOfMasterPAtAcceptedPosition;
            tree.sampleAtAcceptedPosition = otherTree.sampleAtAcceptedPosition;
        }
    }

    private static boolean withProbability(double probability, KeanuRandom random) {
        return random.nextDouble() < probability;
    }

    public static boolean isNotUTurning(Map<VertexId, DoubleTensor> positionForward,
                                        Map<VertexId, DoubleTensor> positionBackward,
                                        Map<VertexId, DoubleTensor> momentumForward,
                                        Map<VertexId, DoubleTensor> momentumBackward) {
        double forward = 0.0;
        double backward = 0.0;

        for (Map.Entry<VertexId, DoubleTensor> forwardPositionForLatent : positionForward.entrySet()) {

            final VertexId latentId = forwardPositionForLatent.getKey();
            final DoubleTensor forwardMinusBackward = forwardPositionForLatent.getValue().minus(
                positionBackward.get(latentId)
            );

            forward += forwardMinusBackward.times(momentumForward.get(latentId)).sum();
            backward += forwardMinusBackward.timesInPlace(momentumBackward.get(latentId)).sum();
        }

        return (forward >= 0.0) && (backward >= 0.0);
    }

    /**
     * This is meant to be used for tracking a sample while building tree.
     *
     * @param sampleFromVertices take samples from these vertices
     */
    private static Map<VertexId, ?> takeSample(List<? extends Vertex> sampleFromVertices) {
        Map<VertexId, ?> sample = new HashMap<>();
        for (Vertex vertex : sampleFromVertices) {
            putValue(vertex, sample);
        }
        return sample;
    }

    private static <T> void putValue(Vertex<T> vertex, Map<VertexId, ?> target) {
        ((Map<VertexId, T>) target).put(vertex.getId(), vertex.getValue());
    }

    public void continueIfNotUTurning(TreeBuilder otherHalfTree) {
        setShouldContinueFlag(otherHalfTree.getShouldContinueFlag() && TreeBuilder.isNotUTurning(
            getForwardPosition(),
            getBackwardPosition(),
            getForwardMomentum(),
            getBackwardMomentum()
        ));
    }

    public boolean getShouldContinueFlag() {
        return shouldContinueFlag;
    }

    public void setShouldContinueFlag(boolean shouldContinueFlag) {
        this.shouldContinueFlag = shouldContinueFlag;
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

    public Map<VertexId, ?> getSampleAtAcceptedPosition() {
        return sampleAtAcceptedPosition;
    }

    public double getLogOfMasterPAtAcceptedPosition() {
        return logOfMasterPAtAcceptedPosition;
    }

    public Map<VertexId, DoubleTensor> getForwardPosition() {
        return leapfrogForward.getPosition();
    }

    public Map<VertexId, DoubleTensor> getBackwardPosition() {
        return leapfrogBackward.getPosition();
    }

    public Map<VertexId, DoubleTensor> getForwardGradient() {
        return leapfrogForward.getGradient();
    }

    public Map<VertexId, DoubleTensor> getBackwardGradient() {
        return leapfrogBackward.getGradient();
    }

    public Map<VertexId, DoubleTensor> getForwardMomentum() {
        return leapfrogForward.getMomentum();
    }

    public Map<VertexId, DoubleTensor> getBackwardMomentum() {
        return leapfrogBackward.getMomentum();
    }

    public void acceptPositionAndGradient() {
        setForwardPosition(acceptedPosition);
        setBackwardPosition(acceptedPosition);
        setForwardGradient(gradientAtAcceptedPosition);
        setBackwardGradient(gradientAtAcceptedPosition);
    }

    private void setForwardPosition(Map<VertexId, DoubleTensor> forwardPosition) {
        leapfrogForward.setPosition(forwardPosition);
    }

    private void setBackwardPosition(Map<VertexId, DoubleTensor> backwardPosition) {
        leapfrogBackward.setPosition(backwardPosition);
    }

    private void setForwardGradient(Map<VertexId, DoubleTensor> forwardGradient) {
        leapfrogForward.setGradient(forwardGradient);
    }

    private void setBackwardGradient(Map<VertexId, DoubleTensor> backwardGradient) {
        leapfrogBackward.setGradient(backwardGradient);
    }

}
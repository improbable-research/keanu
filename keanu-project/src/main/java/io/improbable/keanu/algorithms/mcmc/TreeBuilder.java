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

public class TreeBuilder {

    private static final double DELTA_MAX = 1000.0;

    Leapfrog leapfrogForward;
    Leapfrog leapfrogBackward;
    Map<VertexId, DoubleTensor> acceptedPosition;
    Map<VertexId, DoubleTensor> gradientAtAcceptedPosition;
    double logOfMasterPAtAcceptedPosition;
    Map<VertexId, ?> sampleAtAcceptedPosition;
    int acceptedLeapfrogCount;
    boolean shouldContinueFlag;
    double deltaLikelihoodOfLeapfrog;
    double treeSize;

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
            1,
            true,
            0,
            1
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

            return TreeBuilderBaseCase(latentVertices,
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

                tree.shouldContinueFlag = otherHalfTree.shouldContinueFlag && isNotUTurning(
                    tree.leapfrogForward.position,
                    tree.leapfrogBackward.position,
                    tree.leapfrogForward.momentum,
                    tree.leapfrogBackward.momentum
                );

                tree.acceptedLeapfrogCount += otherHalfTree.acceptedLeapfrogCount;
                tree.deltaLikelihoodOfLeapfrog += otherHalfTree.deltaLikelihoodOfLeapfrog;
                tree.treeSize += otherHalfTree.treeSize;
            }

            return tree;
        }

    }

    public static TreeBuilder TreeBuilderBaseCase(List<Vertex<DoubleTensor>> latentVertices,
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
            leapfrog.position,
            leapfrog.gradient,
            logOfMasterPAfterLeapfrog,
            sampleAtAcceptedPosition,
            acceptedLeapfrogCount,
            shouldContinueFlag,
            deltaLikelihoodOfLeapfrog,
            1
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

}
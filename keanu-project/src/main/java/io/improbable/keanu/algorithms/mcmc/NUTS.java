package io.improbable.keanu.algorithms.mcmc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradient;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

/**
 * Algorithm 6: "No-U-Turn Sampler with Dual Averaging".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
@Builder
public class NUTS implements PosteriorSamplingAlgorithm {

    private static final int DEFAULT_ADAPT_COUNT = 1000;
    private static final double DEFAULT_TARGET_ACCEPTANCE_PROB = 0.65;

    private static final double DELTA_MAX = 1000.0;
    private static final double STABILISER = 10;
    private static final double SHRINKAGE_FACTOR = 0.05;
    private static final double TEND_TO_ZERO_EXPONENT = 0.75;

    public static NUTS withDefaultConfig() {
        return withDefaultConfig(KeanuRandom.getDefaultRandom());
    }

    public static NUTS withDefaultConfig(KeanuRandom random) {
        return NUTS.builder()
            .random(random)
            .build();
    }

    @Getter
    @Setter
    @Builder.Default
    private KeanuRandom random = KeanuRandom.getDefaultRandom();

    //The number of samples for which the stepsize will be tuned. For the remaining samples
    //in which it is not tuned, the stepsize will be frozen to its last calculated value
    @Getter
    @Setter
    @Builder.Default
    private int adaptCount = DEFAULT_ADAPT_COUNT;

    //The target acceptance probability, a suggested value of this is 0.65,
    //Beskos et al., 2010; Neal, 2011
    @Getter
    @Setter
    @Builder.Default
    private double targetAcceptanceProb = DEFAULT_TARGET_ACCEPTANCE_PROB;

    /**
     * Sample from the posterior of a Bayesian Network using the No-U-Turn-Sampling algorithm
     *
     * @param bayesNet           the bayesian network to sample from
     * @param sampleFromVertices the vertices inside the bayesNet to sample from
     * @param sampleCount        the number of samples to take
     * @return Samples taken with NUTS
     */
    @Override
    public NetworkSamples getPosteriorSamples(final BayesianNetwork bayesNet,
                                              final List<? extends Vertex> sampleFromVertices,
                                              final int sampleCount) {

        bayesNet.cascadeObservations();

        final List<Vertex<DoubleTensor>> latentVertices = bayesNet.getContinuousLatentVertices();
        List<? extends Probabilistic> probabilisticVertices = Probabilistic.keepOnlyProbabilisticVertices(bayesNet.getLatentAndObservedVertices());

        final Map<VertexId, List<?>> samples = new HashMap<>();
        addSampleFromCache(samples, takeSample(sampleFromVertices));

        Map<VertexId, DoubleTensor> position = new HashMap<>();
        cachePosition(latentVertices, position);

        Map<VertexId, DoubleTensor> gradient = LogProbGradient.getJointLogProbGradientWrtLatents(probabilisticVertices);

        Map<VertexId, DoubleTensor> momentum = new HashMap<>();

        double initialLogOfMasterP = getLogProb(probabilisticVertices);
        final List<Double> logProbs = new ArrayList<>();
        logProbs.add(initialLogOfMasterP);

        double stepSize = findStartingStepSize(position,
            gradient,
            latentVertices,
            probabilisticVertices,
            random
        );

        AutoTune autoTune = new AutoTune(stepSize,
            targetAcceptanceProb,
            Math.log(stepSize),
            adaptCount
        );

        BuiltTree tree = new BuiltTree(
            position,
            gradient,
            momentum,
            position,
            gradient,
            momentum,
            position,
            gradient,
            initialLogOfMasterP,
            takeSample(sampleFromVertices),
            1,
            true,
            0,
            1
        );

        for (int sampleNum = 1; sampleNum < sampleCount; sampleNum++) {

            initializeMomentumForEachVertex(latentVertices, tree.momentumForward, random);
            cache(tree.momentumForward, tree.momentumBackward);

            double u = random.nextDouble() * Math.exp(tree.logOfMasterPAtAcceptedPosition - 0.5 * dotProduct(tree.momentumForward));

            int treeHeight = 0;
            tree.shouldContinueFlag = true;
            tree.acceptedLeapfrogCount = 1;

            while (tree.shouldContinueFlag) {

                //build tree direction -1 = backwards OR 1 = forwards
                int buildDirection = random.nextBoolean() ? 1 : -1;

                BuiltTree otherHalfTree = buildOtherHalfOfTree(
                    tree,
                    latentVertices,
                    probabilisticVertices,
                    sampleFromVertices,
                    u,
                    buildDirection,
                    treeHeight,
                    stepSize,
                    random
                );

                if (otherHalfTree.shouldContinueFlag) {
                    final double acceptanceProb = (double) otherHalfTree.acceptedLeapfrogCount / tree.acceptedLeapfrogCount;

                    acceptOtherPositionWithProbability(
                        acceptanceProb,
                        tree, otherHalfTree,
                        random
                    );
                }

                tree.acceptedLeapfrogCount += otherHalfTree.acceptedLeapfrogCount;

                tree.deltaLikelihoodOfLeapfrog = otherHalfTree.deltaLikelihoodOfLeapfrog;
                tree.treeSize = otherHalfTree.treeSize;

                tree.shouldContinueFlag = otherHalfTree.shouldContinueFlag && isNotUTurning(
                    tree.positionForward,
                    tree.positionBackward,
                    tree.momentumForward,
                    tree.momentumBackward
                );

                treeHeight++;
            }

            stepSize = adaptStepSize(autoTune, tree, sampleNum);

            tree.positionForward = tree.acceptedPosition;
            tree.gradientForward = tree.gradientAtAcceptedPosition;
            tree.positionBackward = tree.acceptedPosition;
            tree.gradientBackward = tree.gradientAtAcceptedPosition;

            addSampleFromCache(samples, tree.sampleAtAcceptedPosition);
            logProbs.add(tree.logOfMasterPAtAcceptedPosition);
        }

        return new NetworkSamples(samples, logProbs, sampleCount);
    }

    private static BuiltTree buildOtherHalfOfTree(BuiltTree currentTree,
                                                  List<Vertex<DoubleTensor>> latentVertices,
                                                  List<? extends Probabilistic> probabilisticVertices,
                                                  final List<? extends Vertex> sampleFromVertices,
                                                  double u,
                                                  int buildDirection,
                                                  int treeHeight,
                                                  double epsilon,
                                                  KeanuRandom random) {

        BuiltTree otherHalfTree;

        final double logOfMasterPBeforeLeapfrog = getLogProb(probabilisticVertices);
        final double logOfMasterPMinusMomentumBeforeLeapfrog = logOfMasterPBeforeLeapfrog - 0.5 * dotProduct(currentTree.momentumBackward);

        if (buildDirection == -1) {

            otherHalfTree = buildTree(
                latentVertices,
                probabilisticVertices,
                sampleFromVertices,
                currentTree.positionBackward,
                currentTree.gradientBackward,
                currentTree.momentumBackward,
                u,
                buildDirection,
                treeHeight,
                epsilon,
                logOfMasterPMinusMomentumBeforeLeapfrog,
                random
            );

            currentTree.positionBackward = otherHalfTree.positionBackward;
            currentTree.momentumBackward = otherHalfTree.momentumBackward;
            currentTree.gradientBackward = otherHalfTree.gradientBackward;

        } else {

            otherHalfTree = buildTree(
                latentVertices,
                probabilisticVertices,
                sampleFromVertices,
                currentTree.positionForward,
                currentTree.gradientForward,
                currentTree.momentumForward,
                u,
                buildDirection,
                treeHeight,
                epsilon,
                logOfMasterPMinusMomentumBeforeLeapfrog,
                random
            );

            currentTree.positionForward = otherHalfTree.positionForward;
            currentTree.momentumForward = otherHalfTree.momentumForward;
            currentTree.gradientForward = otherHalfTree.gradientForward;
        }

        return otherHalfTree;
    }

    private static BuiltTree buildTree(List<Vertex<DoubleTensor>> latentVertices,
                                       List<? extends Probabilistic> probabilisticVertices,
                                       final List<? extends Vertex> sampleFromVertices,
                                       Map<VertexId, DoubleTensor> position,
                                       Map<VertexId, DoubleTensor> gradient,
                                       Map<VertexId, DoubleTensor> momentum,
                                       double u,
                                       int buildDirection,
                                       int treeHeight,
                                       double epsilon,
                                       double logOfMasterPMinusMomentumBeforeLeapfrog,
                                       KeanuRandom random) {
        if (treeHeight == 0) {

            //Base case-take one leapfrog step in the build direction

            return builtTreeBaseCase(latentVertices,
                probabilisticVertices,
                sampleFromVertices,
                position,
                gradient,
                momentum,
                u,
                buildDirection,
                epsilon,
                logOfMasterPMinusMomentumBeforeLeapfrog
            );

        } else {
            //Recursion-implicitly build the left and right subtrees.

            BuiltTree tree = buildTree(
                latentVertices,
                probabilisticVertices,
                sampleFromVertices,
                position,
                gradient,
                momentum,
                u,
                buildDirection,
                treeHeight - 1,
                epsilon,
                logOfMasterPMinusMomentumBeforeLeapfrog,
                random
            );

            //Should continue building other half if first half's shouldContinueFlag is true
            if (tree.shouldContinueFlag) {

                BuiltTree otherHalfTree = buildOtherHalfOfTree(
                    tree,
                    latentVertices,
                    probabilisticVertices,
                    sampleFromVertices,
                    u,
                    buildDirection,
                    treeHeight - 1,
                    epsilon,
                    random
                );

                double acceptOtherTreePositionProbability = (double) otherHalfTree.acceptedLeapfrogCount / (tree.acceptedLeapfrogCount + otherHalfTree.acceptedLeapfrogCount);

                acceptOtherPositionWithProbability(
                    acceptOtherTreePositionProbability,
                    tree, otherHalfTree,
                    random
                );

                tree.shouldContinueFlag = otherHalfTree.shouldContinueFlag && isNotUTurning(
                    tree.positionForward,
                    tree.positionBackward,
                    tree.momentumForward,
                    tree.momentumBackward
                );

                tree.acceptedLeapfrogCount += otherHalfTree.acceptedLeapfrogCount;
                tree.deltaLikelihoodOfLeapfrog += otherHalfTree.deltaLikelihoodOfLeapfrog;
                tree.treeSize += otherHalfTree.treeSize;
            }

            return tree;
        }

    }

    private static BuiltTree builtTreeBaseCase(List<Vertex<DoubleTensor>> latentVertices,
                                               List<? extends Probabilistic> probabilisticVertices,
                                               final List<? extends Vertex> sampleFromVertices,
                                               Map<VertexId, DoubleTensor> position,
                                               Map<VertexId, DoubleTensor> gradient,
                                               Map<VertexId, DoubleTensor> momentum,
                                               double u,
                                               int buildDirection,
                                               double epsilon,
                                               double logOfMasterPMinusMomentumBeforeLeapfrog) {

        LeapFrogged leapfrog = leapfrog(
            latentVertices,
            probabilisticVertices,
            position,
            gradient,
            momentum,
            epsilon * buildDirection
        );

        final double logOfMasterPAfterLeapfrog = getLogProb(probabilisticVertices);

        final double logOfMasterPMinusMomentum = logOfMasterPAfterLeapfrog - 0.5 * dotProduct(leapfrog.momentum);
        final int acceptedLeapfrogCount = u <= Math.exp(logOfMasterPMinusMomentum) ? 1 : 0;
        final boolean shouldContinueFlag = u < Math.exp(DELTA_MAX + logOfMasterPMinusMomentum);

        final Map<VertexId, ?> sampleAtAcceptedPosition = takeSample(sampleFromVertices);

        double deltaLikelihoodOfLeapfrog = Math.exp(logOfMasterPMinusMomentum - logOfMasterPMinusMomentumBeforeLeapfrog);
        deltaLikelihoodOfLeapfrog = deltaLikelihoodOfLeapfrog < 1 ? deltaLikelihoodOfLeapfrog : 1;

        return new BuiltTree(
            leapfrog.position,
            leapfrog.gradient,
            leapfrog.momentum,
            leapfrog.position,
            leapfrog.gradient,
            leapfrog.momentum,
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

    private static double getLogProb(List<? extends Probabilistic> probabilisticVertices) {
        double sum = 0.0;
        for (Probabilistic vertex : probabilisticVertices) {
            sum += vertex.logProbAtValue();
        }
        return sum;
    }

    private static void acceptOtherPositionWithProbability(double probability,
                                                           BuiltTree tree,
                                                           BuiltTree otherTree,
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

    private static boolean isNotUTurning(Map<VertexId, DoubleTensor> positionForward,
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

    private static void cachePosition(List<Vertex<DoubleTensor>> latentVertices, Map<VertexId, DoubleTensor> position) {
        for (Vertex<DoubleTensor> vertex : latentVertices) {
            position.put(vertex.getId(), vertex.getValue());
        }
    }

    private static void initializeMomentumForEachVertex(List<Vertex<DoubleTensor>> vertices,
                                                        Map<VertexId, DoubleTensor> momentums,
                                                        KeanuRandom random) {
        for (Vertex<DoubleTensor> vertex : vertices) {
            momentums.put(vertex.getId(), random.nextGaussian(vertex.getShape()));
        }
    }

    private static void cache(Map<VertexId, DoubleTensor> from, Map<VertexId, DoubleTensor> to) {
        for (Map.Entry<VertexId, DoubleTensor> entry : from.entrySet()) {
            to.put(entry.getKey(), entry.getValue());
        }
    }

    private static LeapFrogged leapfrog(final List<Vertex<DoubleTensor>> latentVertices,
                                        final List<? extends Probabilistic> probabilisticVertices,
                                        final Map<VertexId, DoubleTensor> position,
                                        final Map<VertexId, DoubleTensor> gradient,
                                        final Map<VertexId, DoubleTensor> momentum,
                                        final double epsilon) {

        final double halfTimeStep = epsilon / 2.0;

        Map<VertexId, DoubleTensor> nextMomentum = new HashMap<>();
        Map<VertexId, DoubleTensor> nextPosition = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> rEntry : momentum.entrySet()) {
            final DoubleTensor updatedMomentum = (gradient.get(rEntry.getKey()).times(halfTimeStep)).plusInPlace(rEntry.getValue());
            nextMomentum.put(rEntry.getKey(), updatedMomentum);
        }

        for (Vertex<DoubleTensor> latent : latentVertices) {
            final DoubleTensor nextPositionForLatent = nextMomentum.get(latent.getId()).
                times(halfTimeStep).
                plusInPlace(
                    position.get(latent.getId())
                );
            nextPosition.put(latent.getId(), nextPositionForLatent);
            latent.setValue(nextPositionForLatent);
        }

        VertexValuePropagation.cascadeUpdate(latentVertices);

        Map<VertexId, DoubleTensor> nextPositionGradient = LogProbGradient.getJointLogProbGradientWrtLatents(probabilisticVertices);

        for (Map.Entry<VertexId, DoubleTensor> nextMomentumForLatent : nextMomentum.entrySet()) {
            final DoubleTensor nextNextMomentumForLatent = nextPositionGradient.get(nextMomentumForLatent.getKey()).
                times(halfTimeStep).
                plusInPlace(
                    nextMomentumForLatent.getValue()
                );
            nextMomentum.put(nextMomentumForLatent.getKey(), nextNextMomentumForLatent);
        }

        return new LeapFrogged(nextPosition, nextMomentum, nextPositionGradient);
    }

    private static double dotProduct(Map<VertexId, DoubleTensor> momentums) {
        double dotProduct = 0.0;
        for (DoubleTensor momentum : momentums.values()) {
            dotProduct += momentum.pow(2).sum();
        }
        return dotProduct;
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

    /**
     * This is used to save of the sample from the uniformly chosen acceptedPosition position
     *
     * @param samples      samples taken already
     * @param cachedSample a cached sample from before leapfrog
     */
    private static void addSampleFromCache(Map<VertexId, List<?>> samples, Map<VertexId, ?> cachedSample) {
        for (Map.Entry<VertexId, ?> sampleEntry : cachedSample.entrySet()) {
            addSampleForVertex(sampleEntry.getKey(), sampleEntry.getValue(), samples);
        }
    }

    private static <T> void addSampleForVertex(VertexId id, T value, Map<VertexId, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(id, v -> new ArrayList<T>());
        samplesForVertex.add(value);
    }

    private static class LeapFrogged {
        final Map<VertexId, DoubleTensor> position;
        final Map<VertexId, DoubleTensor> momentum;
        final Map<VertexId, DoubleTensor> gradient;

        LeapFrogged(Map<VertexId, DoubleTensor> position,
                    Map<VertexId, DoubleTensor> momentum,
                    Map<VertexId, DoubleTensor> gradient) {
            this.position = position;
            this.momentum = momentum;
            this.gradient = gradient;
        }
    }

    private static class BuiltTree {

        Map<VertexId, DoubleTensor> positionBackward;
        Map<VertexId, DoubleTensor> gradientBackward;
        Map<VertexId, DoubleTensor> momentumBackward;
        Map<VertexId, DoubleTensor> positionForward;
        Map<VertexId, DoubleTensor> gradientForward;
        Map<VertexId, DoubleTensor> momentumForward;
        Map<VertexId, DoubleTensor> acceptedPosition;
        Map<VertexId, DoubleTensor> gradientAtAcceptedPosition;
        double logOfMasterPAtAcceptedPosition;
        Map<VertexId, ?> sampleAtAcceptedPosition;
        int acceptedLeapfrogCount;
        boolean shouldContinueFlag;
        double deltaLikelihoodOfLeapfrog;
        double treeSize;

        BuiltTree(Map<VertexId, DoubleTensor> positionBackward,
                  Map<VertexId, DoubleTensor> gradientBackward,
                  Map<VertexId, DoubleTensor> momentumBackward,
                  Map<VertexId, DoubleTensor> positionForward,
                  Map<VertexId, DoubleTensor> gradientForward,
                  Map<VertexId, DoubleTensor> momentumForward,
                  Map<VertexId, DoubleTensor> acceptedPosition,
                  Map<VertexId, DoubleTensor> gradientAtAcceptedPosition,
                  double logOfMasterPAtAcceptedPosition,
                  Map<VertexId, ?> sampleAtAcceptedPosition,
                  int acceptedLeapfrogCount,
                  boolean shouldContinueFlag,
                  double deltaLikelihoodOfLeapfrog,
                  double treeSize) {

            this.positionBackward = positionBackward;
            this.gradientBackward = gradientBackward;
            this.momentumBackward = momentumBackward;
            this.positionForward = positionForward;
            this.gradientForward = gradientForward;
            this.momentumForward = momentumForward;
            this.acceptedPosition = acceptedPosition;
            this.gradientAtAcceptedPosition = gradientAtAcceptedPosition;
            this.logOfMasterPAtAcceptedPosition = logOfMasterPAtAcceptedPosition;
            this.sampleAtAcceptedPosition = sampleAtAcceptedPosition;
            this.acceptedLeapfrogCount = acceptedLeapfrogCount;
            this.shouldContinueFlag = shouldContinueFlag;
            this.deltaLikelihoodOfLeapfrog = deltaLikelihoodOfLeapfrog;
            this.treeSize = treeSize;
        }
    }

    private static class AutoTune {

        double stepSize;
        double averageAcceptanceProb;
        double targetAcceptanceProb;
        double logStepSize;
        double logStepSizeFrozen;
        double adaptCount;
        double shrinkageTarget;

        AutoTune(double stepSize, double targetAcceptanceProb, double logStepSize, int adaptCount) {
            this.stepSize = stepSize;
            this.averageAcceptanceProb = 0;
            this.targetAcceptanceProb = targetAcceptanceProb;
            this.logStepSize = logStepSize;
            this.logStepSizeFrozen = Math.log(1);
            this.adaptCount = adaptCount;
            this.shrinkageTarget = Math.log(10 * stepSize);
        }
    }

    private static double findStartingStepSize(Map<VertexId, DoubleTensor> position,
                                               Map<VertexId, DoubleTensor> gradient,
                                               List<Vertex<DoubleTensor>> vertices,
                                               List<? extends Probabilistic> probabilisticVertices,
                                               KeanuRandom random) {
        double stepsize = 1;
        double probBeforeLeapfrog = getLogProb(probabilisticVertices);
        Map<VertexId, DoubleTensor> momentums = new HashMap<>();
        initializeMomentumForEachVertex(vertices, momentums, random);
        leapfrog(vertices, probabilisticVertices, position, gradient, momentums, stepsize);
        double probAfterLeapfrog = getLogProb(probabilisticVertices);
        double likelihoodRatio = probAfterLeapfrog - probBeforeLeapfrog;
        double scalingFactor = likelihoodRatio > Math.log(0.5) ? 1 : -1;

        while (scalingFactor * (likelihoodRatio) > -scalingFactor * Math.log(2)) {
            stepsize = stepsize * Math.pow(2, scalingFactor);
            leapfrog(vertices, probabilisticVertices, position, gradient, momentums, stepsize);
            likelihoodRatio = getLogProb(probabilisticVertices) - probBeforeLeapfrog;
        }

        return stepsize;
    }

    private static double adaptStepSize(AutoTune autoTune, BuiltTree tree, int sampleNum) {
        if (sampleNum <= autoTune.adaptCount) {
            double percentageLeftToTune = (1 / (sampleNum + STABILISER));
            double acceptanceProb = (autoTune.targetAcceptanceProb - (tree.deltaLikelihoodOfLeapfrog / tree.treeSize));
            double proportionalAcceptanceProb = (1 - percentageLeftToTune) * autoTune.averageAcceptanceProb;
            autoTune.averageAcceptanceProb = proportionalAcceptanceProb + (percentageLeftToTune * acceptanceProb);

            double shrunkSampleCount = Math.sqrt(sampleNum) / SHRINKAGE_FACTOR;
            autoTune.logStepSize = autoTune.shrinkageTarget - (shrunkSampleCount * autoTune.averageAcceptanceProb);

            double tendToZero = Math.pow(sampleNum, -TEND_TO_ZERO_EXPONENT);
            double reducedStepSize = tendToZero * autoTune.logStepSize;
            double increasedStepSizeFrozen = (1 - tendToZero) * autoTune.logStepSizeFrozen;
            autoTune.logStepSizeFrozen = reducedStepSize + increasedStepSizeFrozen;

            return Math.exp(autoTune.logStepSize);
        } else {
            return Math.exp(autoTune.logStepSizeFrozen);
        }
    }

}


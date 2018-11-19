package io.improbable.keanu.algorithms.mcmc;

import com.google.common.base.Preconditions;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.backend.LogProbWithSample;
import io.improbable.keanu.backend.ProbabilisticWithGradientGraph;
import io.improbable.keanu.backend.keanu.KeanuGraphConverter;
import io.improbable.keanu.backend.tensorflow.TensorflowGraphConverter;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toMap;

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

    @Getter
    @Setter
    @Builder.Default
    private double maxTreeHeight = 10;

    public static boolean USE_TENSORFLOW = false;

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

        Preconditions.checkArgument(!sampleFromVertices.isEmpty(), "List of vertices to sample from is empty");
        ProbabilisticWithGradientGraph gradientGraph;
        if (USE_TENSORFLOW) {
            gradientGraph = TensorflowGraphConverter.convertWithGradient(bayesNet);
        } else {
            bayesNet.cascadeObservations();
            gradientGraph = KeanuGraphConverter.convertWithGradient(bayesNet);
        }

        final List<String> latentVariables = gradientGraph.getLatentVariables();
        final List<String> sampleFrom = sampleFromVertices.stream()
            .map(Vertex::getUniqueStringReference)
            .collect(Collectors.toList());

        final Map<String, VertexId> vertexByLabel = sampleFromVertices.stream()
            .collect(Collectors.toMap(Vertex::getUniqueStringReference, Vertex::getId));

        LogProbWithSample logProbWithSample = gradientGraph.logProbWithSample(Collections.emptyMap(), sampleFrom);

        final Map<String, List<?>> samples = new HashMap<>();
        addSampleFromCache(samples, logProbWithSample.getSample());

        Map<String, DoubleTensor> position = new HashMap<>();
        cachePosition((Map<String, DoubleTensor>) gradientGraph.getLatentVariablesValues(), position);

        Map<String, DoubleTensor> gradient = gradientGraph.logProbGradients(Collections.emptyMap());

        double initialLogProb = logProbWithSample.getLogProb();

        final List<Double> logProbForEachSample = new ArrayList<>();
        logProbForEachSample.add(initialLogProb);

        double stepSize = findStartingStepSize(
            position,
            gradient,
            latentVariables,
            initialLogProb,
            gradientGraph,
            random
        );

        AutoTune autoTune = new AutoTune(stepSize,
            targetAcceptanceProb,
            Math.log(stepSize),
            adaptCount
        );

        Map<String, DoubleTensor> momentum = new HashMap<>();
        BuiltTree tree = new BuiltTree(
            position,
            gradient,
            momentum,
            position,
            gradient,
            momentum,
            position,
            gradient,
            initialLogProb,
            logProbWithSample.getSample(),
            1,
            true,
            0,
            1
        );

        for (int sampleNum = 1; sampleNum < sampleCount; sampleNum++) {

            initializeMomentumForEachVertex(latentVariables, tree.momentumForward, tree.positionForward, random);
            cache(tree.momentumForward, tree.momentumBackward);

            double logOfMasterPMinusMomentumBeforeLeapfrog = tree.logOfMasterPAtAcceptedPosition - 0.5 * dotProduct(tree.momentumForward);

            double u = random.nextDouble() * Math.exp(logOfMasterPMinusMomentumBeforeLeapfrog);

            int treeHeight = 0;
            tree.shouldContinueFlag = true;
            tree.acceptedLeapfrogCount = 1;

            System.out.println(tree.treeSize);

            while (tree.shouldContinueFlag && treeHeight <= maxTreeHeight) {

                //build tree direction -1 = backwards OR 1 = forwards
                int buildDirection = random.nextBoolean() ? 1 : -1;

                BuiltTree otherHalfTree = buildOtherHalfOfTree(
                    tree,
                    latentVariables,
                    gradientGraph,
                    sampleFrom,
                    u,
                    buildDirection,
                    treeHeight,
                    stepSize,
                    logOfMasterPMinusMomentumBeforeLeapfrog,
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
            logProbForEachSample.add(tree.logOfMasterPAtAcceptedPosition);
        }

        return new NetworkSamples(toVertexId(samples, vertexByLabel), logProbForEachSample, sampleCount);
    }

    private static Map<VertexId, ? extends List> toVertexId(Map<String, ? extends List> byLabel, Map<String, VertexId> lookup) {
        return byLabel.entrySet().stream()
            .collect(toMap(
                v -> lookup.get(v.getKey()),
                Map.Entry::getValue)
            );
    }

    private static BuiltTree buildOtherHalfOfTree(BuiltTree currentTree,
                                                  List<String> latentVertices,
                                                  ProbabilisticWithGradientGraph probabilisticWithGradientGraph,
                                                  final List<String> sampleFrom,
                                                  double u,
                                                  int buildDirection,
                                                  int treeHeight,
                                                  double epsilon,
                                                  double logOfMasterPMinusMomentumBeforeLeapfrog,
                                                  KeanuRandom random) {

        BuiltTree otherHalfTree;

        if (buildDirection == -1) {

            otherHalfTree = buildTree(
                latentVertices,
                probabilisticWithGradientGraph,
                sampleFrom,
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
                probabilisticWithGradientGraph,
                sampleFrom,
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

    private static BuiltTree buildTree(List<String> latentVertices,
                                       ProbabilisticWithGradientGraph probabilisticWithGradientGraph,
                                       final List<String> sampleFromVertices,
                                       Map<String, DoubleTensor> position,
                                       Map<String, DoubleTensor> gradient,
                                       Map<String, DoubleTensor> momentum,
                                       double u,
                                       int buildDirection,
                                       int treeHeight,
                                       double epsilon,
                                       double logOfMasterPMinusMomentumBeforeLeapfrog,
                                       KeanuRandom random) {
        if (treeHeight == 0) {

            //Base case-take one leapfrog step in the build direction

            return builtTreeBaseCase(latentVertices,
                probabilisticWithGradientGraph,
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
                probabilisticWithGradientGraph,
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
                    probabilisticWithGradientGraph,
                    sampleFromVertices,
                    u,
                    buildDirection,
                    treeHeight - 1,
                    epsilon,
                    logOfMasterPMinusMomentumBeforeLeapfrog,
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

    private static BuiltTree builtTreeBaseCase(List<String> latentVertices,
                                               ProbabilisticWithGradientGraph probabilisticWithGradientGraph,
                                               final List<String> sampleFromVertices,
                                               Map<String, DoubleTensor> position,
                                               Map<String, DoubleTensor> gradient,
                                               Map<String, DoubleTensor> momentum,
                                               double u,
                                               int buildDirection,
                                               double epsilon,
                                               double logOfMasterPMinusMomentumBeforeLeapfrog) {

        LeapFrogged leapfrog = leapfrog(
            latentVertices,
            probabilisticWithGradientGraph,
            position,
            gradient,
            momentum,
            epsilon * buildDirection
        );

        final LogProbWithSample logProbWithSample = probabilisticWithGradientGraph.logProbWithSample(leapfrog.position, sampleFromVertices);
        final double logProbAfterLeapfrog = logProbWithSample.getLogProb();
        final Map<String, ?> sampleAtAcceptedPosition = logProbWithSample.getSample();

        final double logProbMinusMomentum = logProbAfterLeapfrog - 0.5 * dotProduct(leapfrog.momentum);
        final int acceptedLeapfrogCount = u <= Math.exp(logProbMinusMomentum) ? 1 : 0;
        final boolean shouldContinueFlag = u < Math.exp(DELTA_MAX + logProbMinusMomentum);

        final double deltaLikelihoodOfLeapfrog = Math.min(
            1.0,
            Math.exp(logProbMinusMomentum - logOfMasterPMinusMomentumBeforeLeapfrog)
        );

        return new BuiltTree(
            leapfrog.position,
            leapfrog.gradient,
            leapfrog.momentum,
            leapfrog.position,
            leapfrog.gradient,
            leapfrog.momentum,
            leapfrog.position,
            leapfrog.gradient,
            logProbAfterLeapfrog,
            sampleAtAcceptedPosition,
            acceptedLeapfrogCount,
            shouldContinueFlag,
            deltaLikelihoodOfLeapfrog,
            1
        );
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

    private static boolean isNotUTurning(Map<String, DoubleTensor> positionForward,
                                         Map<String, DoubleTensor> positionBackward,
                                         Map<String, DoubleTensor> momentumForward,
                                         Map<String, DoubleTensor> momentumBackward) {
        double forward = 0.0;
        double backward = 0.0;

        for (Map.Entry<String, DoubleTensor> forwardPositionForLatent : positionForward.entrySet()) {

            final String latentId = forwardPositionForLatent.getKey();
            final DoubleTensor forwardMinusBackward = forwardPositionForLatent.getValue().minus(
                positionBackward.get(latentId)
            );

            forward += forwardMinusBackward.times(momentumForward.get(latentId)).sum();
            backward += forwardMinusBackward.timesInPlace(momentumBackward.get(latentId)).sum();
        }

        return (forward >= 0.0) && (backward >= 0.0);
    }

    private static void cachePosition(Map<String, DoubleTensor> latentVertices, Map<String, DoubleTensor> position) {
        for (Map.Entry<String, DoubleTensor> latent : latentVertices.entrySet()) {
            position.put(latent.getKey(), latent.getValue());
        }
    }

    private static void initializeMomentumForEachVertex(List<String> vertices,
                                                        Map<String, DoubleTensor> momentums,
                                                        Map<String, DoubleTensor> positions,
                                                        KeanuRandom random) {
        for (String vertex : vertices) {
            DoubleTensor position = positions.get(vertex);
            momentums.put(vertex, random.nextGaussian(position.getShape()));
        }
    }

    private static void cache(Map<String, DoubleTensor> from, Map<String, DoubleTensor> to) {
        for (Map.Entry<String, DoubleTensor> entry : from.entrySet()) {
            to.put(entry.getKey(), entry.getValue());
        }
    }

    private static LeapFrogged leapfrog(final List<String> latentVertices,
                                        final ProbabilisticWithGradientGraph probabilisticWithGradientGraph,
                                        final Map<String, DoubleTensor> position,
                                        final Map<String, DoubleTensor> gradient,
                                        final Map<String, DoubleTensor> momentum,
                                        final double epsilon) {

        final double halfTimeStep = epsilon / 2.0;

        Map<String, DoubleTensor> nextMomentum = new HashMap<>();
        for (Map.Entry<String, DoubleTensor> momentumForLatent : momentum.entrySet()) {

            final DoubleTensor updatedMomentum = gradient.get(momentumForLatent.getKey())
                .times(halfTimeStep)
                .plusInPlace(momentumForLatent.getValue());

            nextMomentum.put(momentumForLatent.getKey(), updatedMomentum);
        }

        Map<String, DoubleTensor> nextPosition = new HashMap<>();
        for (String latent : latentVertices) {

            final DoubleTensor nextPositionForLatent = nextMomentum.get(latent).
                times(halfTimeStep).
                plusInPlace(
                    position.get(latent)
                );

            nextPosition.put(latent, nextPositionForLatent);
        }

        Map<String, DoubleTensor> nextPositionGradient = probabilisticWithGradientGraph.logProbGradients(nextPosition);

        for (Map.Entry<String, DoubleTensor> nextMomentumForLatent : nextMomentum.entrySet()) {

            final DoubleTensor nextNextMomentumForLatent = nextPositionGradient.get(nextMomentumForLatent.getKey()).
                times(halfTimeStep).
                plusInPlace(nextMomentumForLatent.getValue());

            nextMomentum.put(nextMomentumForLatent.getKey(), nextNextMomentumForLatent);
        }

        return new LeapFrogged(nextPosition, nextMomentum, nextPositionGradient);
    }

    private static double dotProduct(Map<String, DoubleTensor> momentums) {
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
    private static void addSampleFromCache(Map<String, List<?>> samples, Map<String, ?> cachedSample) {
        for (Map.Entry<String, ?> sampleEntry : cachedSample.entrySet()) {
            addSampleForVertex(sampleEntry.getKey(), sampleEntry.getValue(), samples);
        }
    }

    private static <T> void addSampleForVertex(String id, T value, Map<String, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(id, v -> new ArrayList<T>());
        samplesForVertex.add(value);
    }

    private static class LeapFrogged {
        final Map<String, DoubleTensor> position;
        final Map<String, DoubleTensor> momentum;
        final Map<String, DoubleTensor> gradient;

        LeapFrogged(Map<String, DoubleTensor> position,
                    Map<String, DoubleTensor> momentum,
                    Map<String, DoubleTensor> gradient) {
            this.position = position;
            this.momentum = momentum;
            this.gradient = gradient;
        }
    }

    private static class BuiltTree {

        Map<String, DoubleTensor> positionBackward;
        Map<String, DoubleTensor> gradientBackward;
        Map<String, DoubleTensor> momentumBackward;
        Map<String, DoubleTensor> positionForward;
        Map<String, DoubleTensor> gradientForward;
        Map<String, DoubleTensor> momentumForward;
        Map<String, DoubleTensor> acceptedPosition;
        Map<String, DoubleTensor> gradientAtAcceptedPosition;
        double logOfMasterPAtAcceptedPosition;
        Map<String, ?> sampleAtAcceptedPosition;
        int acceptedLeapfrogCount;
        boolean shouldContinueFlag;
        double deltaLikelihoodOfLeapfrog;
        double treeSize;

        BuiltTree(Map<String, DoubleTensor> positionBackward,
                  Map<String, DoubleTensor> gradientBackward,
                  Map<String, DoubleTensor> momentumBackward,
                  Map<String, DoubleTensor> positionForward,
                  Map<String, DoubleTensor> gradientForward,
                  Map<String, DoubleTensor> momentumForward,
                  Map<String, DoubleTensor> acceptedPosition,
                  Map<String, DoubleTensor> gradientAtAcceptedPosition,
                  double logProbAtAcceptedPosition,
                  Map<String, ?> sampleAtAcceptedPosition,
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
            this.logOfMasterPAtAcceptedPosition = logProbAtAcceptedPosition;
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

    private static double findStartingStepSize(Map<String, DoubleTensor> position,
                                               Map<String, DoubleTensor> gradient,
                                               List<String> latentVariables,
                                               double logProbBeforeLeapfrog,
                                               ProbabilisticWithGradientGraph probabilisticGraph,
                                               KeanuRandom random) {
        double stepsize = 1;
        Map<String, DoubleTensor> momentums = new HashMap<>();
        initializeMomentumForEachVertex(latentVariables, momentums, position, random);

        double pThetaR = logProbBeforeLeapfrog - 0.5 * dotProduct(momentums);

        LeapFrogged initialLeapFrog = leapfrog(latentVariables, probabilisticGraph, position, gradient, momentums, stepsize);

        double probAfterLeapfrog = probabilisticGraph.logProb(initialLeapFrog.position);
        double pThetaRAfterLeapFrog = probAfterLeapfrog - 0.5 * dotProduct(initialLeapFrog.momentum);

        double logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
        double scalingFactor = logLikelihoodRatio > Math.log(0.5) ? 1 : -1;

        while (scalingFactor * logLikelihoodRatio > -scalingFactor * Math.log(2)) {
            stepsize = stepsize * Math.pow(2, scalingFactor);

            LeapFrogged leapFrogged = leapfrog(latentVariables, probabilisticGraph, position, gradient, momentums, stepsize);
            probAfterLeapfrog = probabilisticGraph.logProb(leapFrogged.position);
            pThetaRAfterLeapFrog = probAfterLeapfrog - 0.5 * dotProduct(leapFrogged.momentum);

            logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
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

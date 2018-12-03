package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Algorithm 6: "No-U-Turn Sampler with Dual Averaging".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
public class NUTSSampler implements SamplingAlgorithm {

    private static final double DELTA_MAX = 1000.0;
    private static final double STABILISER = 10;
    private static final double SHRINKAGE_FACTOR = 0.05;
    private static final double TEND_TO_ZERO_EXPONENT = 0.75;

    private final KeanuRandom random;
    private final List<Vertex<DoubleTensor>> latentVertices;
    private final List<? extends Vertex> sampleFromVertices;
    private final List<Vertex> probabilisticVertices;
    private final int maxTreeHeight;
    private final boolean adaptEnabled;
    private final AutoTune autoTune;
    private final BuiltTree tree;
    private final LogProbGradientCalculator logProbGradientCalculator;

    private Double stepSize;
    private int sampleNum;

    /**
     * @param sampleFromVertices        vertices to sample from
     * @param latentVertices            vertices that represent latent variables
     * @param probabilisticVertices     vertices that contribute to total log probability (i.e. latent + observed)
     * @param logProbGradientCalculator gradient calculator for diff of log prob with respect to latents
     * @param adaptEnabled              enable the NUTS step size adaptation
     * @param autoTune                  configuration for tuning the stepsize, if adaptEnabled
     * @param tree                      initial tree that will contain the state of the tree build
     * @param stepSize                  The initial step size. A heuristic will be used to determine a suitable initial stepsize if none
     *                                  is given.
     * @param maxTreeHeight             The largest tree height before stopping the hamilitonian process
     * @param random                    the source of randomness
     */
    public NUTSSampler(List<? extends Vertex> sampleFromVertices,
                       List<Vertex<DoubleTensor>> latentVertices,
                       List<Vertex> probabilisticVertices,
                       LogProbGradientCalculator logProbGradientCalculator,
                       boolean adaptEnabled,
                       AutoTune autoTune,
                       BuiltTree tree,
                       Double stepSize,
                       int maxTreeHeight,
                       KeanuRandom random) {

        this.sampleFromVertices = sampleFromVertices;
        this.probabilisticVertices = probabilisticVertices;
        this.latentVertices = latentVertices;
        this.logProbGradientCalculator = logProbGradientCalculator;

        this.sampleNum = 1;
        this.stepSize = stepSize;
        this.tree = tree;
        this.autoTune = autoTune;
        this.maxTreeHeight = maxTreeHeight;
        this.adaptEnabled = adaptEnabled;

        this.random = random;
    }

    @Override
    public void sample(Map<VertexId, List<?>> samples, List<Double> logOfMasterPForEachSample) {
        step();
        addSampleFromCache(samples, tree.sampleAtAcceptedPosition);
        logOfMasterPForEachSample.add(tree.logOfMasterPAtAcceptedPosition);
    }

    @Override
    public NetworkState sample() {
        step();
        return new SimpleNetworkState(tree.sampleAtAcceptedPosition, tree.logOfMasterPAtAcceptedPosition);
    }

    @Override
    public void step() {

        initializeMomentumForEachVertex(latentVertices, tree.momentumForward, random);
        cache(tree.momentumForward, tree.momentumBackward);

        double logOfMasterPMinusMomentumBeforeLeapfrog = tree.logOfMasterPAtAcceptedPosition - 0.5 * dotProduct(tree.momentumForward);

        double logU = Math.log(random.nextDouble()) + logOfMasterPMinusMomentumBeforeLeapfrog;

        int treeHeight = 0;
        tree.shouldContinueFlag = true;
        tree.acceptedLeapfrogCount = 1;

        while (tree.shouldContinueFlag && treeHeight < maxTreeHeight) {

            //build tree direction -1 = backwards OR 1 = forwards
            int buildDirection = random.nextBoolean() ? 1 : -1;

            BuiltTree otherHalfTree = buildOtherHalfOfTree(
                tree,
                latentVertices,
                probabilisticVertices,
                logProbGradientCalculator,
                sampleFromVertices,
                logU,
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

        if (this.adaptEnabled) {
            stepSize = adaptStepSize(autoTune, tree, sampleNum);
        }

        tree.positionForward = tree.acceptedPosition;
        tree.gradientForward = tree.gradientAtAcceptedPosition;
        tree.positionBackward = tree.acceptedPosition;
        tree.gradientBackward = tree.gradientAtAcceptedPosition;

        sampleNum++;
    }

    private static BuiltTree buildOtherHalfOfTree(BuiltTree currentTree,
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

        BuiltTree otherHalfTree;

        if (buildDirection == -1) {

            otherHalfTree = buildTree(
                latentVertices,
                probabilisticVertices,
                logProbGradientCalculator,
                sampleFromVertices,
                currentTree.positionBackward,
                currentTree.gradientBackward,
                currentTree.momentumBackward,
                logU,
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
                logProbGradientCalculator,
                sampleFromVertices,
                currentTree.positionForward,
                currentTree.gradientForward,
                currentTree.momentumForward,
                logU,
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
                                       List<Vertex> probabilisticVertices,
                                       LogProbGradientCalculator logProbGradientCalculator,
                                       final List<? extends Vertex> sampleFromVertices,
                                       Map<VertexId, DoubleTensor> position,
                                       Map<VertexId, DoubleTensor> gradient,
                                       Map<VertexId, DoubleTensor> momentum,
                                       double logU,
                                       int buildDirection,
                                       int treeHeight,
                                       double epsilon,
                                       double logOfMasterPMinusMomentumBeforeLeapfrog,
                                       KeanuRandom random) {
        if (treeHeight == 0) {

            //Base case-take one leapfrog step in the build direction

            return builtTreeBaseCase(latentVertices,
                probabilisticVertices,
                logProbGradientCalculator,
                sampleFromVertices,
                position,
                gradient,
                momentum,
                logU,
                buildDirection,
                epsilon,
                logOfMasterPMinusMomentumBeforeLeapfrog
            );

        } else {
            //Recursion-implicitly build the left and right subtrees.

            BuiltTree tree = buildTree(
                latentVertices,
                probabilisticVertices,
                logProbGradientCalculator,
                sampleFromVertices,
                position,
                gradient,
                momentum,
                logU,
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
                                               List<Vertex> probabilisticVertices,
                                               LogProbGradientCalculator logProbGradientCalculator,
                                               final List<? extends Vertex> sampleFromVertices,
                                               Map<VertexId, DoubleTensor> position,
                                               Map<VertexId, DoubleTensor> gradient,
                                               Map<VertexId, DoubleTensor> momentum,
                                               double logU,
                                               int buildDirection,
                                               double epsilon,
                                               double logOfMasterPMinusMomentumBeforeLeapfrog) {

        LeapFrogged leapfrog = leapfrog(
            latentVertices,
            logProbGradientCalculator,
            position,
            gradient,
            momentum,
            epsilon * buildDirection
        );

        final double logOfMasterPAfterLeapfrog = ProbabilityCalculator.calculateLogProbFor(probabilisticVertices);

        final double logOfMasterPMinusMomentum = logOfMasterPAfterLeapfrog - 0.5 * dotProduct(leapfrog.momentum);
        final int acceptedLeapfrogCount = logU <= logOfMasterPMinusMomentum ? 1 : 0;
        final boolean shouldContinueFlag = logU < DELTA_MAX + logOfMasterPMinusMomentum;

        final Map<VertexId, ?> sampleAtAcceptedPosition = takeSample(sampleFromVertices);

        final double deltaLikelihoodOfLeapfrog = Math.min(
            1.0,
            Math.exp(logOfMasterPMinusMomentum - logOfMasterPMinusMomentumBeforeLeapfrog)
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
            logOfMasterPAfterLeapfrog,
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
                                        final LogProbGradientCalculator logProbGradientCalculator,
                                        final Map<VertexId, DoubleTensor> position,
                                        final Map<VertexId, DoubleTensor> gradient,
                                        final Map<VertexId, DoubleTensor> momentum,
                                        final double epsilon) {

        final double halfTimeStep = epsilon / 2.0;

        Map<VertexId, DoubleTensor> nextMomentum = new HashMap<>();
        for (Map.Entry<VertexId, DoubleTensor> rEntry : momentum.entrySet()) {
            final DoubleTensor updatedMomentum = (gradient.get(rEntry.getKey()).times(halfTimeStep)).plusInPlace(rEntry.getValue());
            nextMomentum.put(rEntry.getKey(), updatedMomentum);
        }

        Map<VertexId, DoubleTensor> nextPosition = new HashMap<>();
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

        Map<VertexId, DoubleTensor> nextPositionGradient = logProbGradientCalculator.getJointLogProbGradientWrtLatents();

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

    /**
     * Taken from algorithm 4 in https://arxiv.org/pdf/1111.4246.pdf.
     */
    static double findStartingStepSize(Map<VertexId, DoubleTensor> position,
                                       Map<VertexId, DoubleTensor> gradient,
                                       List<Vertex<DoubleTensor>> vertices,
                                       List<Vertex> probabilisticVertices,
                                       LogProbGradientCalculator logProbGradientCalculator,
                                       double initialLogOfMasterP,
                                       KeanuRandom random) {
        double stepsize = 1;
        Map<VertexId, DoubleTensor> momentums = new HashMap<>();
        initializeMomentumForEachVertex(vertices, momentums, random);

        double pThetaR = initialLogOfMasterP - 0.5 * dotProduct(momentums);

        NUTSSampler.LeapFrogged initialLeapFrog = NUTSSampler.leapfrog(vertices, logProbGradientCalculator, position, gradient, momentums, stepsize);

        double probAfterLeapfrog = ProbabilityCalculator.calculateLogProbFor(probabilisticVertices);
        double pThetaRAfterLeapFrog = probAfterLeapfrog - 0.5 * dotProduct(initialLeapFrog.momentum);

        double logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
        double scalingFactor = logLikelihoodRatio > Math.log(0.5) ? 1 : -1;

        while (scalingFactor * logLikelihoodRatio > -scalingFactor * Math.log(2)) {
            stepsize = stepsize * Math.pow(2, scalingFactor);

            NUTSSampler.LeapFrogged leapfrog = NUTSSampler.leapfrog(vertices, logProbGradientCalculator, position, gradient, momentums, stepsize);
            probAfterLeapfrog = ProbabilityCalculator.calculateLogProbFor(probabilisticVertices);
            pThetaRAfterLeapFrog = probAfterLeapfrog - 0.5 * dotProduct(leapfrog.momentum);

            logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
        }

        return stepsize;
    }

    /**
     * Taken from algorithm 5 in https://arxiv.org/pdf/1111.4246.pdf.
     */
    private static double adaptStepSize(AutoTune autoTune, BuiltTree tree, int sampleNum) {

        if (sampleNum <= autoTune.adaptCount) {

            //1/(m+t0)
            double percentageLeftToTune = (1 / (sampleNum + STABILISER));

            //(1 - 1/(m+t0)) * Hm-1
            double proportionalAcceptanceProb = (1 - percentageLeftToTune) * autoTune.averageAcceptanceProb;

            //alpha/nu_alpha
            double averageDeltaLikelihoodLeapfrog = tree.deltaLikelihoodOfLeapfrog / tree.treeSize;

            //delta - alpha/nu_alpha
            double acceptanceProb = autoTune.targetAcceptanceProb - averageDeltaLikelihoodLeapfrog;

            //Hm = (1-1/(m+t0)) * Hm-1 + (1/(m+t0)) * (delta - (alpha/nu_alpha))
            double updatedAverageAcceptanceProb = proportionalAcceptanceProb + (percentageLeftToTune * acceptanceProb);

            //sqrt(mu)/gamma
            double shrunkSampleCount = Math.sqrt(sampleNum) / SHRINKAGE_FACTOR;

            //log(epsilon_m) = mu - (sqrt(m)/gamma) * Hm
            double updatedLogStepSize = autoTune.shrinkageTarget - (shrunkSampleCount * updatedAverageAcceptanceProb);

            //m^-k
            double tendToZero = Math.pow(sampleNum, -TEND_TO_ZERO_EXPONENT);

            //m^-k * log(epsilon_m)
            double reducedStepSize = tendToZero * updatedLogStepSize;

            //(1-m^-k) * log(epsilon_bar_m-1)
            double increasedStepSizeFrozen = (1 - tendToZero) * autoTune.logStepSizeFrozen;

            //log(epsilon_bar_m) = m^-k * log(epsilon_m) + (1 - m^-k) * log(epsilon_bar_m-1)
            autoTune.logStepSizeFrozen = reducedStepSize + increasedStepSizeFrozen;

            autoTune.averageAcceptanceProb = updatedAverageAcceptanceProb;
            autoTune.logStepSize = updatedLogStepSize;

            return Math.exp(autoTune.logStepSize);
        } else {

            return Math.exp(autoTune.logStepSizeFrozen);
        }
    }

    static class BuiltTree {

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
                  double logProbAtAcceptedPosition,
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
            this.logOfMasterPAtAcceptedPosition = logProbAtAcceptedPosition;
            this.sampleAtAcceptedPosition = sampleAtAcceptedPosition;
            this.acceptedLeapfrogCount = acceptedLeapfrogCount;
            this.shouldContinueFlag = shouldContinueFlag;
            this.deltaLikelihoodOfLeapfrog = deltaLikelihoodOfLeapfrog;
            this.treeSize = treeSize;
        }
    }

    static class AutoTune {

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
}

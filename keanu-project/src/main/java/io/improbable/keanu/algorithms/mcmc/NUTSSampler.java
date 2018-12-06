package io.improbable.keanu.algorithms.mcmc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

/**
 * Algorithm 6: "No-U-Turn Sampler with Dual Averaging".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
public class NUTSSampler implements SamplingAlgorithm {

    private final KeanuRandom random;
    private final List<Vertex<DoubleTensor>> latentVertices;
    private final List<? extends Vertex> sampleFromVertices;
    private final List<Vertex> probabilisticVertices;
    private final int maxTreeHeight;
    private final boolean adaptEnabled;
    private final AutoTune autoTune;
    private final TreeBuilder tree;
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
                       TreeBuilder tree,
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
        return new SimpleNetworkState(tree.sampleAtAcceptedPosition);
    }

    @Override
    public void step() {

        initializeMomentumForEachVertex(latentVertices, tree.leapfrogForward.momentum, random);
        cache(tree.leapfrogForward.momentum, tree.leapfrogBackward.momentum);

        double logOfMasterPMinusMomentumBeforeLeapfrog = tree.logOfMasterPAtAcceptedPosition - 0.5 * dotProduct(tree.leapfrogForward.momentum);

        double logU = Math.log(random.nextDouble()) + logOfMasterPMinusMomentumBeforeLeapfrog;

        int treeHeight = 0;
        tree.shouldContinueFlag = true;
        tree.acceptedLeapfrogCount = 1;

        while (tree.shouldContinueFlag && treeHeight < maxTreeHeight) {

            //build tree direction -1 = backwards OR 1 = forwards
            int buildDirection = random.nextBoolean() ? 1 : -1;

            TreeBuilder otherHalfTree = TreeBuilder.buildOtherHalfOfTree(
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

                TreeBuilder.acceptOtherPositionWithProbability(
                    acceptanceProb,
                    tree, otherHalfTree,
                    random
                );
            }

            tree.acceptedLeapfrogCount += otherHalfTree.acceptedLeapfrogCount;

            tree.deltaLikelihoodOfLeapfrog = otherHalfTree.deltaLikelihoodOfLeapfrog;
            tree.treeSize = otherHalfTree.treeSize;

            tree.shouldContinueFlag = otherHalfTree.shouldContinueFlag && TreeBuilder.isNotUTurning(
                tree.leapfrogForward.position,
                tree.leapfrogBackward.position,
                tree.leapfrogForward.momentum,
                tree.leapfrogBackward.momentum
            );

            treeHeight++;
        }

        if (this.adaptEnabled) {
            stepSize = autoTune.adaptStepSize(tree, sampleNum);
        }

        tree.leapfrogForward.position = tree.acceptedPosition;
        tree.leapfrogForward.gradient = tree.gradientAtAcceptedPosition;
        tree.leapfrogBackward.position = tree.acceptedPosition;
        tree.leapfrogBackward.gradient = tree.gradientAtAcceptedPosition;

        sampleNum++;
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

    private static double dotProduct(Map<VertexId, DoubleTensor> momentums) {
        double dotProduct = 0.0;
        for (DoubleTensor momentum : momentums.values()) {
            dotProduct += momentum.pow(2).sum();
        }
        return dotProduct;
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

        Leapfrog leapfrog = new Leapfrog(position, momentums, gradient);
        double pThetaR = initialLogOfMasterP - leapfrog.halfDotProductMomentum();

        Leapfrog delta = leapfrog.step(vertices, logProbGradientCalculator, stepsize);

        double probAfterLeapfrog = ProbabilityCalculator.calculateLogProbFor(probabilisticVertices);
        double pThetaRAfterLeapFrog = probAfterLeapfrog - delta.halfDotProductMomentum();

        double logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
        double scalingFactor = logLikelihoodRatio > Math.log(0.5) ? 1 : -1;

        while (scalingFactor * logLikelihoodRatio > -scalingFactor * Math.log(2)) {
            stepsize = stepsize * Math.pow(2, scalingFactor);

            delta = leapfrog.step(vertices, logProbGradientCalculator, stepsize);
            probAfterLeapfrog = ProbabilityCalculator.calculateLogProbFor(probabilisticVertices);
            pThetaRAfterLeapFrog = probAfterLeapfrog - delta.halfDotProductMomentum();

            logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
        }

        return stepsize;
    }

}

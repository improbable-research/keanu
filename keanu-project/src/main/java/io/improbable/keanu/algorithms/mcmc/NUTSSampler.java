package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.sun.javafx.font.Metrics;

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
    private final Stepsize stepsize;
    private final TreeBuilder tree;
    private final LogProbGradientCalculator logProbGradientCalculator;
    private final Statistics statistics;
    private final boolean saveStatistics;
    private int sampleNum;

    /**
     * @param sampleFromVertices        vertices to sample from
     * @param latentVertices            vertices that represent latent variables
     * @param probabilisticVertices     vertices that contribute to total log probability (i.e. latent + observed)
     * @param logProbGradientCalculator gradient calculator for diff of log prob with respect to latents
     * @param adaptEnabled              enable the NUTS step size adaptation
     * @param stepsize                  configuration for tuning the stepsize, if adaptEnabled
     * @param tree                      initial tree that will contain the state of the tree build
     * @param maxTreeHeight             The largest tree height before stopping the hamilitonian process
     * @param random                    the source of randomness
     * @param statistics                the sampler statistics
     * @param saveStatistics            whether to record statistics
     */
    public NUTSSampler(List<? extends Vertex> sampleFromVertices,
                       List<Vertex<DoubleTensor>> latentVertices,
                       List<Vertex> probabilisticVertices,
                       LogProbGradientCalculator logProbGradientCalculator,
                       boolean adaptEnabled,
                       Stepsize stepsize,
                       TreeBuilder tree,
                       int maxTreeHeight,
                       KeanuRandom random,
                       Statistics statistics,
                       boolean saveStatistics) {

        this.sampleFromVertices = sampleFromVertices;
        this.probabilisticVertices = probabilisticVertices;
        this.latentVertices = latentVertices;
        this.logProbGradientCalculator = logProbGradientCalculator;

        this.tree = tree;
        this.stepsize = stepsize;
        this.maxTreeHeight = maxTreeHeight;
        this.adaptEnabled = adaptEnabled;

        this.random = random;
        this.statistics = statistics;
        this.saveStatistics = saveStatistics;

        this.sampleNum = 1;
    }

    @Override
    public void sample(Map<VertexId, List<?>> samples, List<Double> logOfMasterPForEachSample) {
        step();
        addSampleFromCache(samples, tree.getSampleAtAcceptedPosition());
        logOfMasterPForEachSample.add(tree.getLogOfMasterPAtAcceptedPosition());
    }

    @Override
    public NetworkState sample() {
        step();
        return new SimpleNetworkState(tree.getSampleAtAcceptedPosition());
    }

    @Override
    public void step() {

        initializeMomentumForEachVertex(latentVertices, tree.getForwardMomentum(), random);
        cache(tree.getForwardMomentum(), tree.getBackwardMomentum());

        double logOfMasterPMinusMomentumBeforeLeapfrog = tree.getLogOfMasterPAtAcceptedPosition() - 0.5 * dotProduct(tree.getForwardMomentum());

        double logU = Math.log(random.nextDouble()) + logOfMasterPMinusMomentumBeforeLeapfrog;

        int treeHeight = 0;
        tree.resetTreeBeforeSample();

        while (tree.getShouldContinueFlag() && treeHeight < maxTreeHeight) {

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
                stepsize.getStepsize(),
                logOfMasterPMinusMomentumBeforeLeapfrog,
                random
            );

            if (otherHalfTree.getShouldContinueFlag()) {
                final double acceptanceProb = (double) otherHalfTree.getAcceptedLeapfrogCount() / tree.getAcceptedLeapfrogCount();

                TreeBuilder.acceptOtherPositionWithProbability(
                    acceptanceProb,
                    tree,
                    otherHalfTree,
                    random
                );
            }

            tree.incrementLeapfrogCount(otherHalfTree.getAcceptedLeapfrogCount());
            tree.setDeltaLikelihoodOfLeapfrog(otherHalfTree.getDeltaLikelihoodOfLeapfrog());
            tree.setTreeSize(otherHalfTree.getTreeSize());
            tree.continueIfNotUTurning(otherHalfTree);

            treeHeight++;
        }

        if (saveStatistics) {
            recordSamplerStatistics();
        }

        if (this.adaptEnabled) {
            stepsize.adaptStepSize(tree, sampleNum);
        }

        tree.acceptPositionAndGradient();
        sampleNum++;
    }

    private void recordSamplerStatistics() {
        stepsize.recordStatistics(statistics);
        tree.recordStatistics(statistics);
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

}

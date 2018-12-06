package io.improbable.keanu.algorithms.mcmc;

import com.google.common.base.Preconditions;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Algorithm 6: "No-U-Turn Sampler with Dual Averaging".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
@Builder
public class NUTS implements PosteriorSamplingAlgorithm {

    private static final int DEFAULT_ADAPT_COUNT = 1000;
    private static final double DEFAULT_TARGET_ACCEPTANCE_PROB = 0.8;

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

    //The number of samples for which the step size will be tuned. For the remaining samples
    //in which it is not tuned, the step size will be frozen to its last calculated value
    @Getter
    @Setter
    @Builder.Default
    private int adaptCount = DEFAULT_ADAPT_COUNT;

    //Determines whether the step size will adapt during the first adaptCount samples
    @Getter
    @Setter
    @Builder.Default
    private boolean adaptEnabled = true;

    //Sets the initial step size. If none is given then a heuristic will be used to determine a good step size.
    @Getter
    @Setter
    @Builder.Default
    private Double initialStepSize = null;

    //The target acceptance probability, a suggested value of this is 0.65,
    //Beskos et al., 2010; Neal, 2011
    @Getter
    @Setter
    @Builder.Default
    private double targetAcceptanceProb = DEFAULT_TARGET_ACCEPTANCE_PROB;

    //The maximum tree size for the sampler. This controls how long a sample walk can be before it terminates. This
    //will set at a maximum approximately 2^treeSize number of logProb evaluations for a sample.
    @Getter
    @Setter
    @Builder.Default
    private int maxTreeHeight = 10;

    /**
     * Sample from the posterior of a Bayesian Network using the No-U-Turn-Sampling algorithm
     *
     * @param bayesNet           the bayesian network to sample from
     * @param sampleFromVertices the vertices inside the bayesNet to sample from
     * @return Samples taken with NUTS
     */
    public NetworkSamples getPosteriorSamples(final BayesianNetwork bayesNet,
                                              final List<? extends Vertex> sampleFromVertices,
                                              final int sampleCount) {
        return generatePosteriorSamples(bayesNet, sampleFromVertices)
            .generate(sampleCount);
    }

    public NetworkSamples getPosteriorSamples(final BayesianNetwork bayesNet,
                                              final Vertex fromVertex,
                                              final int sampleCount) {
        return getPosteriorSamples(bayesNet, Collections.singletonList(fromVertex), sampleCount);
    }

    public NetworkSamplesGenerator generatePosteriorSamples(final BayesianNetwork bayesNet,
                                                            final List<? extends Vertex> fromVertices) {

        return new NetworkSamplesGenerator(setupSampler(bayesNet, fromVertices), ProgressBar::new);
    }

    private NUTSSampler setupSampler(final BayesianNetwork bayesNet,
                                     final List<? extends Vertex> sampleFromVertices) {

        Preconditions.checkArgument(!sampleFromVertices.isEmpty(), "List of vertices to sample from is empty");
        bayesNet.cascadeObservations();

        final List<Vertex<DoubleTensor>> latentVertices = bayesNet.getContinuousLatentVertices();
        final LogProbGradientCalculator logProbGradientCalculator = new LogProbGradientCalculator(bayesNet.getLatentOrObservedVertices(), latentVertices);
        List<Vertex> probabilisticVertices = bayesNet.getLatentOrObservedVertices();

        Map<VertexId, DoubleTensor> gradient = logProbGradientCalculator.getJointLogProbGradientWrtLatents();
        Map<VertexId, DoubleTensor> momentum = new HashMap<>();
        Map<VertexId, DoubleTensor> position = new HashMap<>();
        cachePosition(latentVertices, position);

        double initialLogOfMasterP = ProbabilityCalculator.calculateLogProbFor(probabilisticVertices);



        double startingStepSize = (initialStepSize == null) ? Stepsize.findStartingStepSize(position,
            gradient,
            latentVertices,
            probabilisticVertices,
            logProbGradientCalculator,
            initialLogOfMasterP,
            random
        ) : initialStepSize;

        Stepsize stepsize = new Stepsize(
            startingStepSize,
            targetAcceptanceProb,
            adaptCount
        );

        TreeBuilder tree = TreeBuilder.createBasicTree(position, momentum, gradient, initialLogOfMasterP, takeSample(sampleFromVertices));

        return new NUTSSampler(
            sampleFromVertices,
            latentVertices,
            probabilisticVertices,
            logProbGradientCalculator,
            adaptEnabled,
            stepsize,
            tree,
            maxTreeHeight,
            random
        );
    }

    private static void cachePosition(List<Vertex<DoubleTensor>> latentVertices, Map<VertexId, DoubleTensor> position) {
        for (Vertex<DoubleTensor> vertex : latentVertices) {
            position.put(vertex.getId(), vertex.getValue());
        }
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

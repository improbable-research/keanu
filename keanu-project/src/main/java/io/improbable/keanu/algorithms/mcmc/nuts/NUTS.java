package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.base.Preconditions;
import com.sun.javafx.font.Metrics;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.mcmc.NetworkSamplesGenerator;
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
import java.util.stream.Collectors;

import static io.improbable.keanu.algorithms.mcmc.SamplingAlgorithm.takeSample;

/**
 * Algorithm 6: "No-U-Turn Sampler with Dual Averaging".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
@Builder
public class NUTS implements PosteriorSamplingAlgorithm {

    private static final int DEFAULT_ADAPT_COUNT = 1000;
    private static final double DEFAULT_TARGET_ACCEPTANCE_PROB = 0.65;

    private final Statistics statistics = new Statistics(Metrics.values());

    public enum Metrics {
        STEPSIZE, LOG_PROB, MEAN_TREE_ACCEPT, TREE_SIZE
    }

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
    @Builder.Default
    private int adaptCount = DEFAULT_ADAPT_COUNT;

    //The target acceptance probability, a suggested value of this is 0.65,
    //Beskos et al., 2010; Neal, 2011
    @Builder.Default
    @Getter
    private double targetAcceptanceProb = DEFAULT_TARGET_ACCEPTANCE_PROB;

    //Determines whether the step size wil
    // l adapt during the first adaptCount samples
    @Builder.Default
    private boolean adaptEnabled = true;

    //Sets the initial step size. If none is given then a heuristic will be used to determine a good step size.
    @Builder.Default
    private Double initialStepSize = null;

    //The maximum tree size for the sampler. This controls how long a sample walk can be before it terminates. This
    //will set at a maximum approximately 2^treeSize number of logProb evaluations for a sample.
    @Builder.Default
    private int maxTreeHeight = 10;

    //Sets whether or not to save debug STATISTICS. The STATISTICS available are: Step size, Log Prob, Mean Tree Acceptance Prob, Tree Size.
    @Builder.Default
    private boolean saveStatistics = false;

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

        Map<VertexId, DoubleTensor> position = latentVertices.stream().collect(Collectors.toMap(Vertex::getId, Vertex::getValue));
        Map<VertexId, DoubleTensor> momentum = new HashMap<>();
        Map<VertexId, DoubleTensor> gradient = logProbGradientCalculator.getJointLogProbGradientWrtLatents();

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

        resetVertexValue(sampleFromVertices, position);

        Tree tree = Tree.createInitialTree(position, momentum, gradient, initialLogOfMasterP, takeSample(sampleFromVertices));

        return new NUTSSampler(
            sampleFromVertices,
            latentVertices,
            probabilisticVertices,
            logProbGradientCalculator,
            adaptEnabled,
            stepsize,
            tree,
            maxTreeHeight,
            random,
            statistics,
            saveStatistics
        );
    }

    public Statistics getStatistics() {
        return statistics;
    }

    private static void resetVertexValue(List<? extends Vertex> sampleFromVertices, Map<VertexId, DoubleTensor> previousPosition) {
        for (Vertex vertex : sampleFromVertices) {
            vertex.setValue(previousPosition.get(vertex.getId()));
        }
    }

}

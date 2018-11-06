package io.improbable.keanu.algorithms.mcmc;

import static io.improbable.keanu.algorithms.mcmc.HamiltonianSampler.addSampleFromVertices;
import static io.improbable.keanu.algorithms.mcmc.HamiltonianSampler.cachePosition;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Hamiltonian Monte Carlo is a method for obtaining samples from a probability
 * distribution with the introduction of a momentum variable.
 * <p>
 * Algorithm 1: "Hamiltonian Monte Carlo".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
@Builder
public class Hamiltonian implements PosteriorSamplingAlgorithm {

    private static final double DEFAULT_STEP_SIZE = 0.1;
    private static final int DEFAULT_LEAP_FROG_COUNT = 20;

    public static Hamiltonian withDefaultConfig() {
        return withDefaultConfig(KeanuRandom.getDefaultRandom());
    }

    public static Hamiltonian withDefaultConfig(KeanuRandom random) {
        return Hamiltonian.builder()
            .random(random)
            .build();
    }

    @Getter
    @Setter
    @Builder.Default
    private KeanuRandom random = KeanuRandom.getDefaultRandom();

    @Getter
    @Setter
    @Builder.Default
    //the number of times to leapfrog in each sample
    private int leapFrogCount = DEFAULT_LEAP_FROG_COUNT;

    @Getter
    @Setter
    @Builder.Default
    //the amount of distance to move each leapfrog
    private double stepSize = DEFAULT_STEP_SIZE;

    /**
     * Sample from the posterior of a Bayesian Network using the Hamiltonian Monte Carlo algorithm
     *
     * @param bayesNet     The bayesian network to sample from
     * @param fromVertices the vertices to sample from
     * @param sampleCount  the number of samples to take
     * @return Samples taken with Hamiltonian Monte Carlo
     */
    @Override
    public NetworkSamples getPosteriorSamples(final BayesianNetwork bayesNet,
                                              final List<? extends Vertex> fromVertices,
                                              final int sampleCount) {
        return generatePosteriorSamples(bayesNet, fromVertices)
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

    private SamplingAlgorithm setupSampler(final BayesianNetwork bayesNet,
                                           final List<? extends Vertex> fromVertices) {
        bayesNet.cascadeObservations();

        final List<Vertex<DoubleTensor>> latentVertices = bayesNet.getContinuousLatentVertices();
        final LogProbGradientCalculator logProbGradientCalculator = new LogProbGradientCalculator(bayesNet.getLatentOrObservedVertices(), latentVertices);

        final Map<VertexId, List<?>> samples = new HashMap<>();
        addSampleFromVertices(samples, fromVertices);

        Map<VertexId, DoubleTensor> position = new HashMap<>();
        cachePosition(latentVertices, position);
        Map<VertexId, DoubleTensor> positionBeforeLeapfrog = new HashMap<>();

        Map<VertexId, DoubleTensor> gradient = logProbGradientCalculator.getJointLogProbGradientWrtLatents();
        Map<VertexId, DoubleTensor> gradientBeforeLeapfrog = new HashMap<>();

        final Map<VertexId, DoubleTensor> momentum = new HashMap<>();
        final Map<VertexId, DoubleTensor> momentumBeforeLeapfrog = new HashMap<>();

        double logOfMasterPBeforeLeapfrog = bayesNet.getLogOfMasterP();
        final List<Double> logOfMasterPForEachSample = new ArrayList<>();
        logOfMasterPForEachSample.add(logOfMasterPBeforeLeapfrog);

        final Map<VertexId, ?> sampleBeforeLeapfrog = new HashMap<>();

        HamiltonianSampler.VertexState before = new HamiltonianSampler.VertexState(position, gradient, momentum);
        HamiltonianSampler.VertexState after = new HamiltonianSampler.VertexState(positionBeforeLeapfrog, gradientBeforeLeapfrog, momentumBeforeLeapfrog);

        return new HamiltonianSampler(
            latentVertices,
            random,
            fromVertices,
            leapFrogCount,
            stepSize,
            bayesNet,
            before,
            after,
            logProbGradientCalculator,
            sampleBeforeLeapfrog,
            logOfMasterPBeforeLeapfrog
        );
    }

}

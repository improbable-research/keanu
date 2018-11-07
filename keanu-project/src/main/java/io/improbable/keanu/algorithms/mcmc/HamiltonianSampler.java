package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class HamiltonianSampler implements SamplingAlgorithm {

    private final List<Vertex<DoubleTensor>> latentVertices;
    private final KeanuRandom random;
    private final List<? extends Vertex> fromVertices;
    private final int leapFrogCount;
    private final double stepSize;
    private final BayesianNetwork bayesNet;

    private VertexState during;
    private VertexState before;
    private LogProbGradientCalculator logProbGradientCalculator;
    private Map<VertexId, ?> sampleBeforeLeapfrog;
    private double logOfMasterPBeforeLeapfrog;

    public HamiltonianSampler(List<Vertex<DoubleTensor>> latentVertices,
                              KeanuRandom random,
                              List<? extends Vertex> fromVertices,
                              int leapFrogCount,
                              double stepSize,
                              BayesianNetwork bayesNet,
                              VertexState during,
                              VertexState before,
                              LogProbGradientCalculator logProbGradientCalculator) {

        this.latentVertices = latentVertices;
        this.random = random;
        this.fromVertices = fromVertices;
        this.leapFrogCount = leapFrogCount;
        this.stepSize = stepSize;
        this.bayesNet = bayesNet;
        this.during = during;
        this.before = before;
        this.logProbGradientCalculator = logProbGradientCalculator;
        this.sampleBeforeLeapfrog = new HashMap<>();
        this.logOfMasterPBeforeLeapfrog = bayesNet.getLogOfMasterP();
    }

    @Override
    public void step() {
        initializeMomentumForEachVertex(latentVertices, during.momentum, random);
        during.cacheState(before);
        sampleBeforeLeapfrog = putSample(fromVertices);

        for (int leapFrogNum = 0; leapFrogNum < leapFrogCount; leapFrogNum++) {
            during.gradient = leapfrog(
                latentVertices,
                during,
                stepSize,
                logProbGradientCalculator
            );
        }
    }

    @Override
    public void sample(Map<VertexId, List<?>> samples, List<Double> logOfMasterPForEachSample) {
        step();
        addSampleFromCache(samples, acceptOrReject());
        logOfMasterPForEachSample.add(logOfMasterPBeforeLeapfrog);
    }

    @Override
    public NetworkState sample() {
        step();
        return new SimpleNetworkState(acceptOrReject());
    }

    private Map<VertexId, ?> acceptOrReject() {
        final double logOfMasterPAfterLeapfrog = bayesNet.getLogOfMasterP();

        final double likelihoodOfLeapfrog = getLikelihoodOfLeapfrog(
            logOfMasterPAfterLeapfrog,
            logOfMasterPBeforeLeapfrog,
            during.momentum,
            before.momentum
        );

        if (shouldReject(likelihoodOfLeapfrog, random)) {

            //Revert to position and gradient before leapfrog
            Map<VertexId, DoubleTensor> tempSwap = during.position;
            during.position = before.position;
            before.position = tempSwap;

            tempSwap = during.gradient;
            during.gradient = before.gradient;
            before.gradient = tempSwap;

            return sampleBeforeLeapfrog;
        } else {
            logOfMasterPBeforeLeapfrog = logOfMasterPAfterLeapfrog;
            return putSample(fromVertices);
        }
    }

    private static void initializeMomentumForEachVertex(List<Vertex<DoubleTensor>> vertexes,
                                                        Map<VertexId, DoubleTensor> momentums,
                                                        KeanuRandom random) {
        for (Vertex<DoubleTensor> currentVertex : vertexes) {
            momentums.put(currentVertex.getId(), random.nextGaussian(currentVertex.getShape()));
        }
    }

    /**
     * function Leapfrog(T, r)
     * Set `r = r + (eps/2)dTL(T)
     * Set `T = T + r`
     * Set `r = r` + (eps/2)dTL(`T)
     * return `T, r`
     *
     * @param latentVertices
     * @param during          current vertex position, gradient and momentum
     * @param stepSize
     * @param logProbGradient calculator of the logProb gradients
     * @return the gradient at the updated position
     */
    private static Map<VertexId, DoubleTensor> leapfrog(final List<Vertex<DoubleTensor>> latentVertices,
                                                        final VertexState during,
                                                        final double stepSize,
                                                        final LogProbGradientCalculator logProbGradient) {

        final double halfTimeStep = stepSize / 2.0;

        Map<VertexId, DoubleTensor> momentumsAtHalfTimeStep = new HashMap<>();

        //Set `r = r + (eps/2)dTL(T)
        for (Map.Entry<VertexId, DoubleTensor> vertexMomentum : during.momentum.entrySet()) {
            final DoubleTensor updatedMomentum = during.gradient.get(vertexMomentum.getKey()).times(halfTimeStep).plusInPlace(vertexMomentum.getValue());
            momentumsAtHalfTimeStep.put(vertexMomentum.getKey(), updatedMomentum);
        }

        //Set `T = T + `r.
        for (Vertex<DoubleTensor> latent : latentVertices) {
            final DoubleTensor nextPosition = momentumsAtHalfTimeStep.get(latent.getId()).times(halfTimeStep).plusInPlace(during.position.get(latent.getId()));
            during.position.put(latent.getId(), nextPosition);
            latent.setValue(nextPosition);
        }

        VertexValuePropagation.cascadeUpdate(latentVertices);

        //Set `r = `r + (eps/2)dTL(`T)
        Map<VertexId, DoubleTensor> newGradient = logProbGradient.getJointLogProbGradientWrtLatents();

        for (Map.Entry<VertexId, DoubleTensor> halfTimeStepMomentum : momentumsAtHalfTimeStep.entrySet()) {
            final DoubleTensor updatedMomentum = newGradient.get(halfTimeStepMomentum.getKey()).times(halfTimeStep).plusInPlace(halfTimeStepMomentum.getValue());
            during.momentum.put(halfTimeStepMomentum.getKey(), updatedMomentum);
        }

        return newGradient;
    }

    private static double getLikelihoodOfLeapfrog(final double logOfMasterPAfterLeapfrog,
                                                  final double previousLogOfMasterP,
                                                  final Map<VertexId, DoubleTensor> leapfroggedMomentum,
                                                  final Map<VertexId, DoubleTensor> momentumPreviousTimeStep) {

        final double leapFroggedMomentumDotProduct = (0.5 * dotProduct(leapfroggedMomentum));
        final double previousMomentumDotProduct = (0.5 * dotProduct(momentumPreviousTimeStep));

        final double leapFroggedLikelihood = logOfMasterPAfterLeapfrog - leapFroggedMomentumDotProduct;
        final double previousLikelihood = previousLogOfMasterP - previousMomentumDotProduct;

        final double logLikelihoodOfLeapFrog = leapFroggedLikelihood - previousLikelihood;
        final double likelihoodOfLeapfrog = Math.exp(logLikelihoodOfLeapFrog);

        return Math.min(likelihoodOfLeapfrog, 1.0);
    }

    private static boolean shouldReject(double likelihood, KeanuRandom random) {
        return likelihood < random.nextDouble();
    }

    private static double dotProduct(Map<VertexId, DoubleTensor> momentums) {
        return momentums.values().stream()
            .mapToDouble(m -> m.pow(2).sum())
            .sum();
    }

    /**
     * This is meant to be used for caching a pre-leapfrog sample. This sample
     * will be used if the leapfrog is rejected.
     *
     * @param fromVertices
     */
    private static Map<VertexId, Object> putSample(List<? extends Vertex> fromVertices) {
        return fromVertices.stream().collect(Collectors.toMap(Vertex::getId, Vertex::getValue));
    }


    /**
     * This is used when a leapfrog is rejected. At that point the vertices are in a post
     * leapfrog state and a pre-leapfrog sample must be used.
     *
     * @param samples
     * @param cachedSample a cached sample from before leapfrog
     */
    private static void addSampleFromCache(Map<VertexId, List<?>> samples, Map<VertexId, ?> cachedSample) {
        for (Map.Entry<VertexId, ?> sampleEntry : cachedSample.entrySet()) {
            addSampleForVertex(sampleEntry.getKey(), sampleEntry.getValue(), samples);
        }
    }

    static void cachePosition(List<Vertex<DoubleTensor>> latentVertices, Map<VertexId, DoubleTensor> position) {
        for (Vertex<DoubleTensor> vertex : latentVertices) {
            position.put(vertex.getId(), vertex.getValue());
        }
    }

    /**
     * This is used when a leapfrog is accepted. At that point the vertices are in a
     * post leapfrog state.
     *
     * @param fromVertices vertices from which to create and save new sample.
     */
    static void addSampleFromVertices(Map<VertexId, List<?>> samples, List<? extends Vertex> fromVertices) {
        for (Vertex<?> vertex : fromVertices) {
            addSampleForVertex(vertex.getId(), vertex.getValue(), samples);
        }
    }

    static <T> void addSampleForVertex(VertexId id, T value, Map<VertexId, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(id, v -> new ArrayList<T>());
        samplesForVertex.add(value);
    }

    static class VertexState {

        Map<VertexId, DoubleTensor> position;
        Map<VertexId, DoubleTensor> gradient;
        Map<VertexId, DoubleTensor> momentum;

        public VertexState(Map<VertexId, DoubleTensor> position, Map<VertexId, DoubleTensor> gradient) {
            this.position = position;
            this.gradient = gradient;
            this.momentum = new HashMap<>();
        }

        void cacheState(VertexState that) {
            that.position.putAll(position);
            that.gradient.putAll(gradient);
            that.momentum.putAll(momentum);
        }
    }

}

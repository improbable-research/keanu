package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradient;

import java.util.*;

/**
 * Algorithm 3: "Efficient NUTS".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
public class EfficientNUTS {

    private EfficientNUTS() {
    }

    public static NetworkSamples getPosteriorSamples(final BayesNet bayesNet,
                                                     final List<DoubleVertex> fromVertices,
                                                     final int sampleCount,
                                                     final double stepSize) {

        return getPosteriorSamples(bayesNet, fromVertices, sampleCount, stepSize, new Random());
    }

    public static NetworkSamples getPosteriorSamples(final BayesNet bayesNet,
                                                     final List<? extends Vertex<?>> fromVertices,
                                                     final int sampleCount,
                                                     final double epsilon,
                                                     final Random random) {

        final List<Vertex<Double>> latentVertices = bayesNet.getContinuousLatentVertices();
        final List<Vertex<?>> probabilisticVertices = bayesNet.getVerticesThatContributeToMasterP();

        final Map<String, List<?>> samples = new HashMap<>();
        addSampleFromVertices(samples, fromVertices);

        Map<String, Double> positionBeforeLeapfrog = new HashMap<>();
        cachePosition(latentVertices, positionBeforeLeapfrog);

        Map<String, Double> position = new HashMap<>();
        Map<String, Double> positionForward = new HashMap<>();
        Map<String, Double> positionBackward = new HashMap<>();

        Map<String, Double> gradient = LogProbGradient.getJointLogProbGradientWrtLatents(
                bayesNet.getVerticesThatContributeToMasterP()
        );
        Map<String, Double> gradientBeforeLeapfrog = new HashMap<>();

        final Map<String, Double> momentumForward = new HashMap<>();
        final Map<String, Double> momentumBackward = new HashMap<>();
        final Map<String, Double> momentumBeforeLeapfrog = new HashMap<>();

        double logOfMasterPreviously = bayesNet.getLogOfMasterP();

        final Map<String, ?> sampleBeforeLeapfrog = new HashMap<>();

        for (int sampleNum = 1; sampleNum < sampleCount; sampleNum++) {

            cache(positionBeforeLeapfrog, position);
            cache(positionBeforeLeapfrog, positionForward);
            cache(positionBeforeLeapfrog, positionBackward);

            cache(gradient, gradientBeforeLeapfrog);

            initializeMomentumForEachVertex(latentVertices, momentumBeforeLeapfrog, random);
            cache(momentumBeforeLeapfrog, momentumForward);
            cache(momentumBeforeLeapfrog, momentumBackward);

            takeSample(sampleBeforeLeapfrog, fromVertices);

            double u = random.nextDouble() * logOfMasterPreviously - 0.5 * dotProduct(momentumBackward);
            int j = 0;
            int s = 1;
            int n = 1;

            while (s == 1) {
                int v = random.nextBoolean() ? 1 : -1;

                BuiltTree builtTree;
                if (v == -1) {
                    builtTree = BuildTree(
                            bayesNet.getContinuousLatentVertices(),
                            bayesNet.getVerticesThatContributeToMasterP(),
                            positionBackward,
                            gradient,
                            momentumBackward,
                            u,
                            v,
                            j,
                            epsilon,
                            random
                    );
                } else {
                    builtTree = BuildTree(positionForward, momentumForward, u, v, j, epsilon);
                }

                if (builtTree.sPrime == 1) {
                    final double acceptanceProb = n / (double) builtTree.nPrime;
                    if (random.nextDouble() < acceptanceProb) {
                        position = builtTree.thetaPrime;

                        //TODO: update logOfMasterPBeforeLeapfrog
                    }
                }

                n = n + builtTree.nPrime;

                s = builtTree.sPrime * isUTurning(
                        positionForward,
                        positionBackward,
                        momentumForward,
                        momentumBackward
                );

                j++;
            }

        }

        return new NetworkSamples(samples, sampleCount);
    }

    private static double getLogOfMasterP(List<Vertex<?>> probabilisticVertices) {
        double sum = 0.0;
        for (Vertex<?> vertex : probabilisticVertices) {
            sum += vertex.logDensityAtValue();
        }
        return sum;
    }

    private static BuiltTree BuildTree(List<Vertex<Double>> latentVertices,
                                       List<Vertex<?>> probabilisticVertices,
                                       Map<String, Double> position,
                                       Map<String, Double> gradient,
                                       Map<String, Double> momentum,
                                       double u,
                                       int v,
                                       int j,
                                       double epsilon,
                                       Random random
    ) {

        final double deltaMax = 1000.0;
        if (j == 1) {
            //Base case—take one leapfrog step in the direction v

            gradient = leapfrog(
                    latentVertices,
                    probabilisticVertices,
                    position,
                    gradient,
                    momentum,
                    epsilon
            );

            final double logOfMasterPAfterLeapfrog = getLogOfMasterP(probabilisticVertices);

            final double logMpMinusMomentum = logOfMasterPAfterLeapfrog - 0.5 * dotProduct(momentum);

            final int nPrime = withProbability(Math.exp(logMpMinusMomentum), random) ? 1 : 0;
            final int sPrime = withProbability(Math.exp(deltaMax + logMpMinusMomentum), random) ? 1 : 0;

            return new BuiltTree(position, )
        } else {

        }

    }

    private static boolean withProbability(double probability, Random random) {
        return random.nextDouble() < probability;
    }

    private static int isUTurning(Map<String, Double> positionForward,
                                  Map<String, Double> positionBack,
                                  Map<String, Double> momentumForward,
                                  Map<String, Double> momentumBack) {
        double forward = 0.0;
        double backward = 0.0;

        for (Map.Entry<String, Double> forwardEntry : positionForward.entrySet()) {

            final String id = forwardEntry.getKey();
            final double forwardMinusBack = forwardEntry.getValue() - positionBack.get(id);

            forward += forwardMinusBack * momentumForward.get(id);
            backward += forwardMinusBack * momentumBack.get(id);
        }

        boolean turning = forward >= 0.0 || backward >= 0.0;

        return turning ? 1 : 0;
    }

    private static void cachePosition(List<Vertex<Double>> latentVertices, Map<String, Double> position) {
        for (Vertex<Double> vertex : latentVertices) {
            position.put(vertex.getId(), vertex.getValue());
        }
    }

    private static Map<String, Double> initializeMomentumForEachVertex(List<Vertex<Double>> vertexes,
                                                                       Map<String, Double> momentums,
                                                                       Random random) {
        for (int i = 0; i < vertexes.size(); i++) {
            Vertex currentVertex = vertexes.get(i);
            momentums.put(currentVertex.getId(), random.nextGaussian());
        }
        return momentums;
    }

    private static void cache(Map<String, Double> from, Map<String, Double> to) {
        for (Map.Entry<String, Double> entry : from.entrySet()) {
            to.put(entry.getKey(), entry.getValue());
        }
    }

    /**
     * function Leapfrog(θ, r, eps)
     * Set ˜r ← r + (eps/2)∇θL(θ)
     * Set ˜θ ← θ + r˜
     * Set ˜r ← r˜ + (eps/2)∇θL(˜θ)
     * return ˜θ, r˜
     */
    private static Map<String, Double> leapfrog(final List<Vertex<Double>> latentVertices,
                                                final List<Vertex<?>> probabilisticVertices,
                                                final Map<String, Double> position,
                                                final Map<String, Double> gradient,
                                                final Map<String, Double> momentums,
                                                final double epsilon) {

        final double halfTimeStep = epsilon / 2.0;

        Map<String, Double> momentumsAtHalfTimeStep = new HashMap<>();

        //Set ˜r ← r + (eps/2)∇θL(θ)
        for (Map.Entry<String, Double> vertexMomentum : momentums.entrySet()) {
            final double updatedMomentum = vertexMomentum.getValue() + halfTimeStep * gradient.get(vertexMomentum.getKey());
            momentumsAtHalfTimeStep.put(vertexMomentum.getKey(), updatedMomentum);
        }

        //Set ˜θ ← θ + ˜r.
        for (Vertex<Double> latent : latentVertices) {
            final double nextPosition = position.get(latent.getId()) + halfTimeStep * momentumsAtHalfTimeStep.get(latent.getId());
            position.put(latent.getId(), nextPosition);
            latent.setAndCascade(nextPosition);
        }

        //Set ˜r ← ˜r + (eps/2)∇θL(˜θ)
        Map<String, Double> newGradient = LogProbGradient.getJointLogProbGradientWrtLatents(
                probabilisticVertices
        );

        for (Map.Entry<String, Double> halfTimeStepMomentum : momentumsAtHalfTimeStep.entrySet()) {
            final double updatedMomentum = halfTimeStepMomentum.getValue() + halfTimeStep * newGradient.get(halfTimeStepMomentum.getKey());
            momentums.put(halfTimeStepMomentum.getKey(), updatedMomentum);
        }

        return newGradient;
    }

    private static double dotProduct(Map<String, Double> momentums) {
        double dotProduct = 0.0;
        for (Double momentum : momentums.values()) {
            dotProduct += momentum * momentum;
        }
        return dotProduct;
    }

    /**
     * This is meant to be used for caching a pre-leapfrog sample. This sample
     * will be used if the leapfrog is rejected.
     *
     * @param sample
     * @param fromVertices
     */
    private static void takeSample(Map<String, ?> sample, List<? extends Vertex<?>> fromVertices) {
        for (Vertex<?> vertex : fromVertices) {
            putValue(vertex, sample);
        }
    }

    private static <T> void putValue(Vertex<T> vertex, Map<String, ?> target) {
        ((Map<String, T>) target).put(vertex.getId(), vertex.getValue());
    }

    /**
     * This is used when a leapfrog is rejected. At that point the vertices are in a post
     * leapfrog state and a pre-leapfrog sample must be used.
     *
     * @param samples
     * @param cachedSample a cached sample from before leapfrog
     */
    private static void addSampleFromCache(Map<String, List<?>> samples, Map<String, ?> cachedSample) {
        for (Map.Entry<String, ?> sampleEntry : cachedSample.entrySet()) {
            addSampleForVertex(sampleEntry.getKey(), sampleEntry.getValue(), samples);
        }
    }

    /**
     * This is used when a leapfrog is accepted. At that point the vertices are in a
     * post leapfrog state.
     *
     * @param samples
     * @param fromVertices vertices from which to create and save new sample.
     */
    private static void addSampleFromVertices(Map<String, List<?>> samples, List<? extends Vertex<?>> fromVertices) {
        for (Vertex<?> vertex : fromVertices) {
            addSampleForVertex(vertex.getId(), vertex.getValue(), samples);
        }
    }

    private static <T> void addSampleForVertex(String id, T value, Map<String, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(id, v -> new ArrayList<T>());
        samplesForVertex.add(value);
    }

    private static class BuiltTree {

        public final Map<String, Double> positionBackward;
        public final Map<String, Double> momentumBackward;
        public final Map<String, Double> positionForward;
        public final Map<String, Double> momentumForward;
        public final Map<String, Double> thetaPrime;
        public final int nPrime;
        public final int sPrime;

        public BuiltTree(Map<String, Double> positionBackward,
                         Map<String, Double> momentumBackward,
                         Map<String, Double> positionForward,
                         Map<String, Double> momentumForward,
                         Map<String, Double> thetaPrime,
                         int nPrime,
                         int sPrime) {

            this.positionBackward = positionBackward;
            this.momentumBackward = momentumBackward;
            this.positionForward = positionForward;
            this.momentumForward = momentumForward;
            this.thetaPrime = thetaPrime;
            this.nPrime = nPrime;
            this.sPrime = sPrime;
        }

    }

}


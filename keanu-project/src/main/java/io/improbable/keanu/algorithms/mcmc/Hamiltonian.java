package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradient;

import java.util.*;

/**
 * Hamiltonian Monte Carlo is a method for obtaining samples from a probability
 * distribution with the introduction of a momentum variable.
 * <p>
 * Algorithm 1: "Hamiltonian Monte Carlo".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
public class Hamiltonian {

    private Hamiltonian() {
    }

    public static NetworkSamples getPosteriorSamples(final BayesNet bayesNet,
                                                     final List<DoubleVertex> fromVertices,
                                                     final int sampleCount,
                                                     final int leapFrogCount,
                                                     final double stepSize) {

        return getPosteriorSamples(bayesNet, fromVertices, sampleCount, leapFrogCount, stepSize, new Random());
    }

    public static NetworkSamples getPosteriorSamples(final BayesNet bayesNet,
                                                     final List<? extends Vertex<?>> fromVertices,
                                                     final int sampleCount,
                                                     final int leapFrogCount,
                                                     final double stepSize,
                                                     final Random random) {

        final List<Vertex<Double>> latentVertices = bayesNet.getContinuousLatentVertices();
        final List<Vertex<?>> probabilisticVertices = bayesNet.getVerticesThatContributeToMasterP();

        final Map<String, List<?>> samples = new HashMap<>();
        takeSamples(samples, fromVertices);

        final Map<String, Double> positionBeforeLeapfrog = new HashMap<>();
        final Map<String, Double> momentum = new HashMap<>();
        final Map<String, Double> momentumBeforeLeapfrog = new HashMap<>();

        for (int sampleNum = 1; sampleNum < sampleCount; sampleNum++) {

            cachePosition(latentVertices, positionBeforeLeapfrog);

            initializeMomentumForEachVertex(latentVertices, momentum, random);
            cacheMomentum(momentum, momentumBeforeLeapfrog);

            final double logOfMasterPBeforeLeapfrog = bayesNet.getLogOfMasterP();

            Map<String, Double> gradient = LogProbGradient
                    .getJointLogProbGradientWrtLatents(bayesNet.getVerticesThatContributeToMasterP());

            for (int leapFrogNum = 0; leapFrogNum < leapFrogCount; leapFrogNum++) {
                gradient = leapfrog(
                        latentVertices,
                        gradient,
                        momentum,
                        stepSize,
                        probabilisticVertices
                );
            }

            final double likelihoodOfLeapfrog = getLikelihoodOfLeapfrog(
                    bayesNet,
                    momentum,
                    logOfMasterPBeforeLeapfrog,
                    momentumBeforeLeapfrog
            );

            if (shouldReject(likelihoodOfLeapfrog, random)) {
                revertToPosition(latentVertices, positionBeforeLeapfrog);
            }

            takeSamples(samples, fromVertices);
        }

        return new NetworkSamples(samples, sampleCount);
    }

    private static Map<String, Double> cachePosition(List<Vertex<Double>> latentVertices, Map<String, Double> position) {
        for (Vertex<Double> vertex : latentVertices) {
            position.put(vertex.getId(), vertex.getValue());
        }
        return position;
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

    private static void cacheMomentum(Map<String, Double> momentums, Map<String, Double> cache) {
        for (Map.Entry<String, Double> entry : momentums.entrySet()) {
            cache.put(entry.getKey(), entry.getValue());
        }
    }

    /**
     * function Leapfrog(θ, r, )
     * Set ˜r ← r + (eps/2)∇θL(θ)
     * Set ˜θ ← θ + r˜
     * Set ˜r ← r˜ + (eps/2)∇θL(˜θ)
     * return ˜θ, r˜
     *
     * @param latentVertices
     * @param gradient              gradient at current position
     * @param momentums             current vertex momentums
     * @param stepSize
     * @param probabilisticVertices all vertices that impact the joint posterior (masterP)
     * @return the gradient at the updated position
     */
    private static Map<String, Double> leapfrog(final List<Vertex<Double>> latentVertices,
                                                final Map<String, Double> gradient,
                                                final Map<String, Double> momentums,
                                                final double stepSize,
                                                final List<Vertex<?>> probabilisticVertices) {

        final double halfTimeStep = stepSize / 2.0;

        Map<String, Double> momentumsAtHalfTimeStep = new HashMap<>();

        //Set ˜r ← r + (eps/2)∇θL(θ)
        for (Map.Entry<String, Double> vertexMomentum : momentums.entrySet()) {
            final double updatedMomentum = vertexMomentum.getValue() + halfTimeStep * gradient.get(vertexMomentum.getKey());
            momentumsAtHalfTimeStep.put(vertexMomentum.getKey(), updatedMomentum);
        }

        //Set ˜θ ← θ + ˜r.
        for (Vertex<Double> latent : latentVertices) {
            final double nextPosition = latent.getValue() + halfTimeStep * momentumsAtHalfTimeStep.get(latent.getId());
            latent.setAndCascade(nextPosition);
        }

        //Set ˜r ← ˜r + (eps/2)∇θL(˜θ)
        Map<String, Double> newGradient = LogProbGradient
                .getJointLogProbGradientWrtLatents(probabilisticVertices);

        for (Map.Entry<String, Double> halfTimeStepMomentum : momentumsAtHalfTimeStep.entrySet()) {
            final double updatedMomentum = halfTimeStepMomentum.getValue() + halfTimeStep * newGradient.get(halfTimeStepMomentum.getKey());
            momentums.put(halfTimeStepMomentum.getKey(), updatedMomentum);
        }

        return newGradient;
    }

    private static void revertToPosition(List<Vertex<Double>> continuousLatentVertices,
                                         Map<String, Double> values) {

        for (Vertex<Double> latent : continuousLatentVertices) {
            latent.setAndCascade(values.get(latent.getId()));
        }
    }

    private static double getLikelihoodOfLeapfrog(final BayesNet bayesNet,
                                                  final Map<String, Double> leapfroggedMomentum,
                                                  final double previousLogOfMasterP,
                                                  final Map<String, Double> momentumPreviousTimeStep) {
        final double logOfMasterP = bayesNet.getLogOfMasterP();

        final double leapFroggedMomentumDotProduct = (0.5 * dotProduct(leapfroggedMomentum));
        final double previousMomentumDotProduct = (0.5 * dotProduct(momentumPreviousTimeStep));

        final double leapFroggedLikelihood = logOfMasterP - leapFroggedMomentumDotProduct;
        final double previousLikelihood = previousLogOfMasterP - previousMomentumDotProduct;

        final double logLikelihoodOfLeapFrog = leapFroggedLikelihood - previousLikelihood;
        final double likelihoodOfLeapfrog = Math.exp(logLikelihoodOfLeapFrog);

        return Math.min(likelihoodOfLeapfrog, 1.0);
    }

    private static boolean shouldReject(double likelihood, Random random) {
        return likelihood < random.nextDouble();
    }

    private static double dotProduct(Map<String, Double> momentums) {
        double dotProduct = 0.0;
        for (Double momentum : momentums.values()) {
            dotProduct += momentum * momentum;
        }
        return dotProduct;
    }

    private static void takeSamples(Map<String, List<?>> samples, List<? extends Vertex<?>> fromVertices) {
        for (Vertex<?> vertex : fromVertices) {
            addSampleForVertex(vertex, samples);
        }
    }

    private static <T> void addSampleForVertex(Vertex<T> vertex, Map<String, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<T>());
        samplesForVertex.add(vertex.getValue());
    }

}

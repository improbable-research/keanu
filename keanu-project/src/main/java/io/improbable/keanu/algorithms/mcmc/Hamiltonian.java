package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.variational.FitnessFunctionWithGradient;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

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

        final Map<String, List<?>> samples = new HashMap<>();
        takeSamples(samples, fromVertices);

        final Map<String, Double> positionBeforeLeapfrog = new HashMap<>();

        final Map<String, Double> momentum = new HashMap<>();
        final Map<String, Double> momentumBeforeLeapfrog = new HashMap<>();

        for (int sampleNum = 1; sampleNum < sampleCount; sampleNum++) {

            cachePosition(latentVertices, positionBeforeLeapfrog);

            initializeMomentumForEachVertex(latentVertices, momentum, random);
            cacheMomentum(momentum, momentumBeforeLeapfrog);

            final double logOfMasterP = bayesNet.getLogOfMasterP();

            for (int leapFrogNum = 0; leapFrogNum < leapFrogCount; leapFrogNum++) {
                leapfrogVertices(
                        latentVertices,
                        momentum,
                        stepSize,
                        bayesNet
                );
            }

            final double likelihoodOfLeapfrog = getLikelihoodOfLeapfrog(
                    bayesNet,
                    momentum,
                    logOfMasterP,
                    momentumBeforeLeapfrog
            );

            if (shouldReject(likelihoodOfLeapfrog, random)) {
                //revert position to previous
                setAndCascadeValues(latentVertices, positionBeforeLeapfrog);
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

    private static void leapfrogVertices(final List<Vertex<Double>> latentVertices,
                                         final Map<String, Double> momentum,
                                         final double stepSize,
                                         final BayesNet bayesNet) {

        for (Vertex<Double> currentVertex : latentVertices) {

            final double vertexMomentum = momentum.get(currentVertex.getId());

            final double newMomentum = leapfrogVertex(
                    currentVertex,
                    vertexMomentum,
                    stepSize,
                    bayesNet
            );

            momentum.put(currentVertex.getId(), newMomentum);
        }
    }

    private static double leapfrogVertex(final Vertex<Double> vertex,
                                         final double vertexMomentum,
                                         final double stepSize,
                                         final BayesNet bayesNet) {

        final double halfTimeStep = stepSize / 2.0;

        Map<String, Double> gradients = FitnessFunctionWithGradient
                .getDiffsWithRespectToUpstreamLatents(bayesNet.getVerticesThatContributeToMasterP());

        final double gradient = gradients.get(vertex.getId());

        final double logOfMasterPBeforeLeap = bayesNet.getLogOfMasterP();

        final double momentumHalfTimeStep = vertexMomentum - (halfTimeStep * gradient * logOfMasterPBeforeLeap);

        final double positionTimeStep = vertex.getValue() + (stepSize * momentumHalfTimeStep);

        vertex.setAndCascade(positionTimeStep);

        Map<String, Double> newGradients = FitnessFunctionWithGradient
                .getDiffsWithRespectToUpstreamLatents(bayesNet.getVerticesThatContributeToMasterP());

        final double newGradient = newGradients.get(vertex.getId());

        //TODO: master p only due to change in vertex
        final double logOfMasterPAfterLeap = bayesNet.getLogOfMasterP();

        final double momentumTimeStep = momentumHalfTimeStep - (halfTimeStep * newGradient * logOfMasterPAfterLeap);

        return momentumTimeStep;
    }

    private static void setAndCascadeValues(List<Vertex<Double>> continuousLatentVertices,
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
        fromVertices.forEach(vertex -> addSampleForVertex(vertex, samples));
    }

    private static <T> void addSampleForVertex(Vertex<T> vertex, Map<String, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<T>());
        samplesForVertex.add(vertex.getValue());
    }

}

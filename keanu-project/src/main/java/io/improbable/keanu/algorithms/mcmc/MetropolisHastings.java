package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.graphtraversal.MarkovBlanket;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Metropolis Hastings is a Markov Chain Monte Carlo method for obtaining samples from a probability distribution
 */
public class MetropolisHastings {

    private MetropolisHastings() {
    }

    public static NetworkSamples getPosteriorSamples(BayesianNetwork bayesNet,
                                                     List<? extends Vertex> fromVertices,
                                                     int sampleCount) {
        return getPosteriorSamples(bayesNet, fromVertices, sampleCount, KeanuRandom.getDefaultRandom());
    }

    /**
     * @param bayesNet     a bayesian network containing latent vertices
     * @param fromVertices the vertices to include in the returned samples
     * @param sampleCount  number of samples to take using the algorithm
     * @param random       the source of randomness
     * @return Samples for each vertex ordered by MCMC iteration
     */
    public static NetworkSamples getPosteriorSamples(final BayesianNetwork bayesNet,
                                                     final List<? extends Vertex> fromVertices,
                                                     final int sampleCount,
                                                     final KeanuRandom random) {
        checkBayesNetInHealthyState(bayesNet);

        Map<Long, List<?>> samplesByVertex = new HashMap<>();
        List<Vertex> latentVertices = bayesNet.getLatentVertices();
        Map<Vertex, Set<Vertex>> affectedVerticesCache = getVerticesAffectedByLatents(latentVertices);

        double logP = bayesNet.getLogOfMasterP();
        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {

            Vertex<?> chosenVertex = latentVertices.get(sampleNum % latentVertices.size());
            Set<Vertex> affectedVertices = affectedVerticesCache.get(chosenVertex);
            logP = nextSample(chosenVertex, logP, affectedVertices, 1.0, random);

            takeSamples(samplesByVertex, fromVertices);
        }

        return new NetworkSamples(samplesByVertex, sampleCount);
    }

    static <T> double nextSample(final Vertex<T> chosenVertex,
                                 final double logPOld,
                                 final Set<Vertex> affectedVertices,
                                 final double T,
                                 final KeanuRandom random) {

        final double affectedVerticesLogPOld = sumLogP(affectedVertices);

        final T oldValue = chosenVertex.getValue();
        final T proposedValue = chosenVertex.sample(random);

        chosenVertex.setAndCascade(proposedValue);

        final double affectedVerticesLogPNew = sumLogP(affectedVertices);

        if (affectedVerticesLogPNew != Double.NEGATIVE_INFINITY) {

            final double logPNew = logPOld - affectedVerticesLogPOld + affectedVerticesLogPNew;

            final double pqxOld = chosenVertex.logProb(oldValue);
            final double pqxNew = chosenVertex.logProb(proposedValue);

            final double logr = (logPNew * (1.0 / T) + pqxOld) - (logPOld * (1.0 / T) + pqxNew);
            final double r = Math.exp(logr);

            final boolean shouldAccept = r >= random.nextDouble();

            if (shouldAccept) {
                return logPNew;
            }
        }

        //reject change
        chosenVertex.setAndCascade(oldValue);
        return logPOld;
    }

    static Map<Vertex, Set<Vertex>> getVerticesAffectedByLatents(List<? extends Vertex> latentVertices) {
        return latentVertices.stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> {
                    Set<Vertex> affectedVertices = new HashSet<>();
                    affectedVertices.add(v);
                    affectedVertices.addAll(MarkovBlanket.getDownstreamProbabilisticVertices(v));
                    return affectedVertices;
                }));
    }

    private static double sumLogP(Set<Vertex> vertices) {
        double sum = 0.0;
        for (Vertex v : vertices) {
            sum += v.logProbAtValue();
        }
        return sum;
    }

    private static void takeSamples(Map<Long, List<?>> samples, List<? extends Vertex> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex((Vertex<?>) vertex, samples));
    }

    private static <T> void addSampleForVertex(Vertex<T> vertex, Map<Long, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<T>());
        samplesForVertex.add(vertex.getValue());
    }

    private static void checkBayesNetInHealthyState(BayesianNetwork bayesNet) {
        bayesNet.cascadeObservations();
        if (bayesNet.getLatentAndObservedVertices().isEmpty()) {
            throw new IllegalArgumentException("Cannot sample from a completely deterministic BayesNet");
        } else if (bayesNet.isInImpossibleState()) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }
    }

}

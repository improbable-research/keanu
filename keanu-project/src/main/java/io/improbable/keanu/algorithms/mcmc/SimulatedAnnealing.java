package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.vertices.Vertex;

import java.util.*;

/**
 * Simulated Annealing is a modified version of Metropolis Hastings that causes the MCMC random walk to
 * tend towards the Maximum A Posteriori (MAP)
 */
public class SimulatedAnnealing {

    public static NetworkState getMaxAPosteriori(BayesNet bayesNet,
                                                 int sampleCount,
                                                 AnnealingSchedule schedule) {
        return getMaxAPosteriori(bayesNet, sampleCount, schedule, new Random());
    }

    public static NetworkState getMaxAPosteriori(BayesNet bayesNet,
                                                 int sampleCount,
                                                 Random random) {
        AnnealingSchedule schedule = exponentialSchedule(sampleCount, 2, 0.01);
        return getMaxAPosteriori(bayesNet, sampleCount, schedule, random);
    }

    public static NetworkState getMaxAPosteriori(BayesNet bayesNet, int sampleCount) {

        AnnealingSchedule schedule = exponentialSchedule(sampleCount, 2, 0.01);

        return getMaxAPosteriori(bayesNet, sampleCount, schedule, new Random());
    }

    /**
     * Finds the MAP using the default annealing schedule, which is an exponential decay schedule.
     *
     * @param bayesNet          a bayesian network containing latent vertices
     * @param sampleCount       the number of samples to take
     * @param annealingSchedule the schedule to update T (temperature) as a function of sample number.
     * @param random            the source of randomness
     * @return the NetworkState that represents the Max A Posteriori
     */
    public static NetworkState getMaxAPosteriori(BayesNet bayesNet,
                                                 int sampleCount,
                                                 AnnealingSchedule annealingSchedule,
                                                 Random random) {

        if (bayesNet.isInImpossibleState()) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }

        Map<String, ?> maxSamplesByVertex = new HashMap<>();
        List<Vertex> latentVertices = bayesNet.getLatentVertices();

        Map<Vertex, Set<Vertex>> affectedVerticesCache = MetropolisHastings.getVerticesAffectedByLatents(latentVertices);

        Map<String, Map<String, Long>> setAndCascadeCache = new HashMap<>();

        double logP = bayesNet.getLogOfMasterP();
        double maxLogP = logP;
        setSamplesAsMax(maxSamplesByVertex, latentVertices);

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {

            Vertex<?> chosenVertex = latentVertices.get(sampleNum % latentVertices.size());

            double temperature = annealingSchedule.getTemperature(sampleNum);
            Set<Vertex> affectedVertices = affectedVerticesCache.get(chosenVertex);
            logP = MetropolisHastings.nextSample(chosenVertex, logP, affectedVertices, temperature, setAndCascadeCache, random);

            if (logP > maxLogP) {
                maxLogP = logP;
                setSamplesAsMax(maxSamplesByVertex, latentVertices);
            }
        }

        return new SimpleNetworkState(maxSamplesByVertex);
    }

    private static void setSamplesAsMax(Map<String, ?> samples, List<? extends Vertex> fromVertices) {
        fromVertices.forEach(vertex -> setSampleForVertex((Vertex<?>) vertex, samples));
    }

    private static <T> void setSampleForVertex(Vertex<T> vertex, Map<String, ?> samples) {
        ((Map<String, ? super T>) samples).put(vertex.getId(), vertex.getValue());
    }

    /**
     * An annealing schedule determines how T (temperature) changes as
     * a function of the current iteration number (i.e. sample number)
     */
    public interface AnnealingSchedule {
        double getTemperature(int iteration);
    }

    /**
     * @param iterations the number of iterations annealing over
     * @param startT     the value of T at iteration 0
     * @param endT       the value of T at the last iteration
     * @return the annealing schedule
     */
    public static AnnealingSchedule exponentialSchedule(int iterations, double startT, double endT) {

        final double minusK = Math.log(endT / startT) / iterations;

        return n -> startT * Math.exp(minusK * n);
    }

}

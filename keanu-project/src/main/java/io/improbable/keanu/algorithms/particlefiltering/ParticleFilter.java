package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.*;

import static java.lang.Math.exp;

public class ParticleFilter {

    private ParticleFilter() {
    }

    /***
     * A particle filtering approach is used to find probable values for the latent vertices of a Bayesian network,
     * given a set of observed vertices. This is done by incrementally increasing the proportion of the graph under
     * consideration, randomly sampling values for newly added latent variables at each stage. Each increment
     * involves the addition of one new observed vertex and the latent vertices that it depend on. This is done for a
     * specified number of 'particles', each of which represents one set of randomly sampled values of the latent
     * vertices in the Bayesian network and has an associated probability. As the proportion of the graph under
     * consideration incrementally grows, less probable particles are culled and more probable particles are duplicated,
     * resulting in a final set of relatively probable particles.
     *
     * This methodology is similar to the Sequential Importance Resampling Algorithm described here
     * (https://www.lancaster.ac.uk/pg/turnerl/PartileFiltering.pdf).
     *
     * @param vertices the vertices of a Bayesian network to find probable values for
     * @param numParticles the number of particles to generate (a larger number will yield better results but is more
     *                     computationally expensive)
     * @param resamplingCycles the number of times low probability particles are culled and high probability particles
     *                         are replicated each time the subgraph under consideration is expanded
     * @param resamplingProportion the proportion of particles to cull (e.g. the 25% of least probably particles could
     *                             be culled)
     * @param random a random number generator
     * @return a list of particles representing the most probable found values of latent variables
     */

    public static List<Particle> getProbableValues(Collection<? extends Vertex> vertices, int numParticles,
                                                   int resamplingCycles, double resamplingProportion, KeanuRandom random) {

        Map<Vertex, Set<Vertex>> obsVertIncrDependencies = LatentIncrementSort.sort(vertices);
        List<Vertex> observedVertexOrder = new ArrayList<>(obsVertIncrDependencies.keySet());
        List<Particle> particles = createEmptyParticles(numParticles);

        for (int i = 0; i < observedVertexOrder.size(); i++) {
            Vertex<?> nextObsVertex = observedVertexOrder.get(i);
            Set<Vertex> vertexDeps = obsVertIncrDependencies.get(nextObsVertex);
            particles = updateParticles(nextObsVertex, vertexDeps, particles, numParticles, resamplingCycles,
                resamplingProportion, random);
        }

        return particles;
    }

    private static List<Particle> updateParticles(Vertex<?> nextObservedVertex,
                                                  Set<Vertex> vertexDeps,
                                                  List<Particle> particles,
                                                  int numParticles,
                                                  int resamplingCycles,
                                                  double resamplingProportion,
                                                  KeanuRandom random) {

        List<Particle> updatedParticles = sampleAndCopy(particles, numParticles, random);
        addObservedVertexToParticles(updatedParticles, nextObservedVertex, vertexDeps, random);

        for (int i = 0; i < resamplingCycles; i++) {
            updatedParticles = removeWorstParticles(updatedParticles, resamplingProportion);
            int numToSample = numParticles - updatedParticles.size();
            List<Particle> sampledParticles = sampleAndCopy(particles, numToSample, random);
            addObservedVertexToParticles(sampledParticles, nextObservedVertex, vertexDeps, random);
            updatedParticles.addAll(sampledParticles);
        }

        return updatedParticles;
    }


    private static List<Particle> createEmptyParticles(int number) {

        List<Particle> emptyParticles = new ArrayList<>();
        for (int i = 0; i < number; i++) {
            emptyParticles.add(new Particle());
        }

        return emptyParticles;
    }

    private static void addObservedVertexToParticles(List<Particle> particles,
                                                     Vertex<?> observedVertex,
                                                     Set<Vertex> vertexDependencies,
                                                     KeanuRandom random) {

        for (Particle particle : particles) {
            particle.addObservedVertex(observedVertex);
            for (Vertex<?> latentVertex : vertexDependencies) {
                sampleValueAndAddToParticle(latentVertex, particle, random);
            }

            particle.updateSumLogPOfSubgraph();
        }
    }

    private static <T> void sampleValueAndAddToParticle(Vertex<T> vertex, Particle particle, KeanuRandom random) {
        T sample = vertex.sample(random);
        particle.addLatentVertex(vertex, sample);
    }

    private static List<Particle> removeWorstParticles(List<Particle> particles, double proportionToRemove) {
        particles.sort(Particle::sortDescending);
        int numberToKeep = (int) (particles.size() * (1.0 - proportionToRemove));
        List<Particle> particlesToKeep = particles.subList(0, numberToKeep);
        return new ArrayList<>(particlesToKeep);
    }

    private static List<Particle> sampleAndCopy(List<Particle> particles, int numToSample, KeanuRandom random) {

        double sumWeights = particles.stream().mapToDouble(p -> exp(p.getSumLogPOfSubgraph())).sum();
        List<Particle> sampledParticles = new ArrayList<>();
        for (int i = 0; i < numToSample; i++) {
            Particle sampledParticle = weightedRandomParticle(particles, sumWeights, random);
            sampledParticles.add(sampledParticle.shallowCopy());
        }

        return sampledParticles;
    }

    private static Particle weightedRandomParticle(List<Particle> particles, double sumWeights, KeanuRandom random) {
        double r = random.nextDouble() * sumWeights;
        double cumulativeWeight = 0;
        Particle p = particles.get(0);
        for (int i = 0; i < particles.size(); i++) {
            p = particles.get(i);
            cumulativeWeight += exp(p.getSumLogPOfSubgraph());
            if (cumulativeWeight > r) {
                break;
            }
        }

        return p;
    }

    public static class Particle {

        private Map<Vertex, Object> latentVertices = new HashMap<>();
        private List<Vertex> observedVertices = new ArrayList<>();
        private double sumLogPOfSubgraph = 1.0;

        public Map<Vertex, Object> getLatentVertices() {
            return latentVertices;
        }

        public double getSumLogPOfSubgraph() {
            return sumLogPOfSubgraph;
        }

        public <T> void addLatentVertex(Vertex<T> vertex, T value) {
            latentVertices.put(vertex, value);
        }

        public <T> void addObservedVertex(Vertex<T> vertex) {
            observedVertices.add(vertex);
        }

        public double updateSumLogPOfSubgraph() {
            applyLatentVertexValues();
            double sumLogPOfLatents = sumLogP(latentVertices.keySet());
            double sumLogPOfObservables = sumLogP(observedVertices);
            sumLogPOfSubgraph = sumLogPOfLatents + sumLogPOfObservables;
            return sumLogPOfSubgraph;
        }

        public Particle shallowCopy() {
            Particle clone = new Particle();
            clone.latentVertices = new HashMap<>(this.latentVertices);
            clone.observedVertices = new ArrayList<>(this.observedVertices);
            return clone;
        }

        public static int sortDescending(Particle a, Particle b) {
            return Double.compare(b.getSumLogPOfSubgraph(), a.getSumLogPOfSubgraph());
        }

        private void applyLatentVertexValues() {
            latentVertices.keySet().forEach(this::applyLatentVertexValue);
        }

        private <T> void applyLatentVertexValue(Vertex<T> vertex) {
            if (latentVertices.containsKey(vertex)) {
                T value = (T) latentVertices.get(vertex);
                vertex.setAndCascade(value);
            }
        }

        private double sumLogP(Collection<Vertex> vertices) {
            return vertices.stream().mapToDouble(Vertex::logProbAtValue).sum();
        }
    }
}

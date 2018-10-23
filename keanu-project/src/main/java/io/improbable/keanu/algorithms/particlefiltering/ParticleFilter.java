package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

/***
 * This class allows you to create particle filters to find likely states of a network (i.e. Particles)
 *
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
 */
public class ParticleFilter {
    private Collection<? extends Vertex> vertices;
    private int numParticles;
    private int resamplingCycles;
    private double resamplingProportion;
    private KeanuRandom random;
    private List<Particle> particles;

    public static ParticleFilterBuilder ofVertexInGraph(Vertex vertex) {
        return new ParticleFilterBuilder(vertex.getConnectedGraph());
    }

    /**
     * Creates a Particle Filter and runs the algorithm
     *
     * @param vertices the vertices of a Bayesian network to find probable values for
     * @param numParticles the number of particles to generate (a larger number will yield better results but is more
     *  computationally expensive)
     * @param resamplingCycles the number of times low probability particles are culled and high probability particles
     *                         are replicated each time the subgraph under consideration is expanded
     * @param resamplingProportion the proportion of particles to cull (e.g. the 25% of least probably particles could
     *                             be culled)
     * @param random a random number generator
     * @return a list of particles representing the most probable found values of latent variables
     */
    public ParticleFilter(Collection<? extends Vertex> vertices,
                          int numParticles,
                          int resamplingCycles,
                          double resamplingProportion,
                          KeanuRandom random) {
        this.vertices = vertices;
        this.numParticles = numParticles;
        this.resamplingCycles = resamplingCycles;
        this.resamplingProportion = resamplingProportion;
        this.random = random;
        sort();
    }

    /**
     * @return a list of probable particles (network states) sorted in descending order of likelihood
     */
    public List<Particle> getSortedMostProbableParticles() {
        particles.sort(Particle::sortDescending);
        return particles;
    }

    public Particle getMostProbableParticle() {
        return particles.get(0);
    }

    public List<Particle> getMostProbableParticles() {
        return particles;
    }

    private void sort() {
        Map<Vertex, Set<Vertex>> obsVertIncrDependencies = LatentIncrementSort.sort(this.vertices);
        List<Vertex> observedVertexOrder = new ArrayList<>(obsVertIncrDependencies.keySet());
        List<Particle> particles = createEmptyParticles(this.numParticles);

        for (int i = 0; i < observedVertexOrder.size(); i++) {
            Vertex<?> nextObsVertex = observedVertexOrder.get(i);
            Set<Vertex> vertexDeps = obsVertIncrDependencies.get(nextObsVertex);
            particles = updateParticles(nextObsVertex, vertexDeps, particles);
        }

        this.particles = particles;
    }

    private List<Particle> updateParticles(Vertex<?> nextObservedVertex,
                                                  Set<Vertex> vertexDeps,
                                                  List<Particle> particles) {

        List<Particle> updatedParticles = sampleAndCopy(particles, numParticles);
        addObservedVertexToParticles(updatedParticles, nextObservedVertex, vertexDeps);

        for (int i = 0; i < this.resamplingCycles; i++) {
            updatedParticles = removeWorstParticles(updatedParticles);
            int numToSample = this.numParticles - updatedParticles.size();
            List<Particle> sampledParticles = sampleAndCopy(particles, numToSample);
            addObservedVertexToParticles(sampledParticles, nextObservedVertex, vertexDeps);
            updatedParticles.addAll(sampledParticles);
        }

        return updatedParticles;
    }


    private List<Particle> createEmptyParticles(int number) {

        List<Particle> emptyParticles = new ArrayList<>();
        for (int i = 0; i < number; i++) {
            emptyParticles.add(new Particle());
        }

        return emptyParticles;
    }

    private void addObservedVertexToParticles(List<Particle> particles,
                                                     Vertex<?> observedVertex,
                                                     Set<Vertex> vertexDependencies) {

        for (Particle particle : particles) {
            particle.addObservedVertex(observedVertex);
            for (Vertex<?> latentVertex : vertexDependencies) {
                sampleValueAndAddToParticle(latentVertex, particle);
            }

            particle.updateSumLogPOfSubgraph();
        }
    }

    private <T> void sampleValueAndAddToParticle(Vertex<T> vertex, Particle particle) {
        T sample = vertex.sample(random);
        particle.addLatentVertex(vertex, sample);
    }

    private List<Particle> removeWorstParticles(List<Particle> particles) {
        particles.sort(Particle::sortDescending);
        int numberToKeep = (int) (particles.size() * (1.0 - resamplingProportion));
        List<Particle> particlesToKeep = particles.subList(0, numberToKeep);
        return new ArrayList<>(particlesToKeep);
    }

    private List<Particle> sampleAndCopy(List<Particle> particles, int numToSample) {

        double sumWeights = particles.stream().mapToDouble(Particle::prob).sum();
        List<Particle> sampledParticles = new ArrayList<>();
        for (int i = 0; i < numToSample; i++) {
            Particle sampledParticle = weightedRandomParticle(particles, sumWeights);
            sampledParticles.add(sampledParticle.shallowCopy());
        }

        return sampledParticles;
    }

    private Particle weightedRandomParticle(List<Particle> particles, double sumWeights) {
        double r = random.nextDouble() * sumWeights;
        double cumulativeWeight = 0;
        Particle p = particles.get(0);
        for (int i = 0; i < particles.size(); i++) {
            p = particles.get(i);
            cumulativeWeight += p.prob();
            if (cumulativeWeight > r) {
                break;
            }
        }

        return p;
    }
}

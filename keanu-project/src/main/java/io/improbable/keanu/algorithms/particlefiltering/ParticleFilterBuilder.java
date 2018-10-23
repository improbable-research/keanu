package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Collection;

public class ParticleFilterBuilder {
    private Collection<? extends Vertex> vertices;
    private int numParticles = 1000;
    private int resamplingCycles = 3;
    private double resamplingProportion = 0.5;
    private KeanuRandom random = KeanuRandom.getDefaultRandom();

    /**
     * @param vertices the vertices of a Bayesian network to find probable values for
     */
    public ParticleFilterBuilder(Collection<? extends Vertex> vertices) {
        this.vertices = vertices;
    }

    /**
     * @param numParticles the number of particles to generate (a larger number will yield better results but is more
     *       computationally expensive)
     * @return this
     */
    public ParticleFilterBuilder withNumParticles(int numParticles) {
        this.numParticles = numParticles;
        return this;
    }

    /**
     * @param resamplingCycles the number of times low probability particles are culled and high probability particles
     *                         are replicated each time the subgraph under consideration is expanded
     * @return this
     */
    public ParticleFilterBuilder withResamplingCycles(int resamplingCycles) {
        this.resamplingCycles = resamplingCycles;
        return this;
    }

    /**
     * @param resamplingProportion the proportion of particles to cull (e.g. the 25% of least probably particles could
     *                            be culled)
     * @return this
     */
    public ParticleFilterBuilder withResamplingProportion(double resamplingProportion) {
        this.resamplingProportion = resamplingProportion;
        return this;
    }

    /**
     * @param random A {@link KeanuRandom KeanuRandom} used for stochastic parts of algorithm
     * @return this
     */
    public ParticleFilterBuilder withRandomness(KeanuRandom random) {
        this.random = random;
        return this;
    }

    public ParticleFilter build() {
        return new ParticleFilter(vertices, numParticles, resamplingCycles, resamplingProportion, random);
    }
}

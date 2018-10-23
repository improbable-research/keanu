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

    public ParticleFilterBuilder(Collection<? extends Vertex> vertices) {
        this.vertices = vertices;
    }

    public ParticleFilterBuilder withNumParticles(int numParticles) {
        this.numParticles = numParticles;
        return this;
    }

    public ParticleFilterBuilder withResamplingCycles(int resamplingCycles) {
        this.resamplingCycles = resamplingCycles;
        return this;
    }

    public ParticleFilterBuilder withResamplingProportion(double resamplingProportion) {
        this.resamplingProportion = resamplingProportion;
        return this;
    }

    public ParticleFilterBuilder withRandomness(KeanuRandom random) {
        this.random = random;
        return this;
    }

    public ParticleFilter build() {
        return new ParticleFilter(vertices, numParticles, resamplingCycles, resamplingProportion, random);
    }
}

package io.improbable.snippet;

import io.improbable.keanu.algorithms.particlefiltering.Particle;
import io.improbable.keanu.algorithms.particlefiltering.ParticleFilter;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;

import java.util.List;

public class ParticleFilterExample {
    public static void particleFilterExample() {
        //%%SNIPPET_START%% ParticleFilterExample
        //Create a dummy Bayesian Network
        DoubleVertex temperature = new UniformVertex(0.0, 100.0);
        DoubleVertex noiseAMu = new GaussianVertex(0.0, 2.0);
        DoubleVertex noiseA = new GaussianVertex(noiseAMu, 2.0);
        DoubleVertex noiseBMu = new GaussianVertex(0.0, 2.0);
        DoubleVertex noiseB = new GaussianVertex(noiseBMu, 2.0);
        DoubleVertex thermometerA = new GaussianVertex(temperature.plus(noiseA), 1.0);
        DoubleVertex thermometerB = new GaussianVertex(temperature.plus(noiseB), 1.0);
        thermometerA.observe(21.0);
        thermometerB.observe(19.5);

        //Create a particle filter with default settings
        ParticleFilter filter = ParticleFilter.ofVertexInGraph(temperature)
            .build();

        //Get a sorted list of the most probable particles in order of descending probability
        List<Particle> particles = filter.getSortedMostProbableParticles();

        //Get the most probable particle
        Particle mostProbableParticle = filter.getMostProbableParticle();

        //Get the estimated temperature in the most probable particle
        double estimation = mostProbableParticle.getScalarValueOfVertex(temperature);

        //Get the probability of this particle state
        double logProb = mostProbableParticle.logProb();
        //%%SNIPPET_END%% ParticleFilterExample
    }
}
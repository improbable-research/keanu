package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static java.lang.Math.exp;
import static org.junit.Assert.assertTrue;

public class ParticleFilteringTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    private final Logger log = LoggerFactory.getLogger(ParticleFilteringTest.class);

    @Test
    public void findsCorrectTemp() {

        DoubleVertex temperature = new UniformVertex(0.0, 100.0);
        DoubleVertex noiseAMu = new GaussianVertex(0.0, 2.0);
        DoubleVertex noiseA = new GaussianVertex(noiseAMu, 2.0);
        DoubleVertex noiseBMu = new GaussianVertex(0.0, 2.0);
        DoubleVertex noiseB = new GaussianVertex(noiseBMu, 2.0);
        DoubleVertex noiseCMu = new GaussianVertex(0.0, 2.0);
        DoubleVertex noiseC = new GaussianVertex(noiseCMu, 2.0);
        DoubleVertex noiseDMu = new GaussianVertex(0.0, 2.0);
        DoubleVertex noiseD = new GaussianVertex(noiseDMu, 2.0);
        DoubleVertex thermometerA = new GaussianVertex(temperature.plus(noiseA), 1.0);
        DoubleVertex thermometerB = new GaussianVertex(temperature.plus(noiseB), 1.0);
        DoubleVertex thermometerC = new GaussianVertex(temperature.plus(noiseC), 1.0);
        DoubleVertex thermometerD = new GaussianVertex(temperature.plus(noiseD), 1.0);
        thermometerA.observe(21.0);
        thermometerB.observe(19.5);
        thermometerC.observe(22.0);
        thermometerD.observe(18.0);

        int numParticles = 1000;
        int resamplingCycles = 3;
        double resamplingProportion = 0.5;

        List<ParticleFilter.Particle> particles = ParticleFilter.getProbableValues(
            temperature.getConnectedGraph(),
            numParticles,
            resamplingCycles,
            resamplingProportion,
            new KeanuRandom(1)
        );

        particles.sort(ParticleFilter.Particle::sortDescending);
        ParticleFilter.Particle p = particles.get(0);

        double estimatedTemp = ((DoubleTensor) p.getLatentVertices().get(temperature)).scalar();
        double probability = exp(p.getSumLogPOfSubgraph());

        log.info("Final temp estimate = " + estimatedTemp + ", probability = " + probability);

        assertTrue(estimatedTemp > 18.0);
        assertTrue(estimatedTemp < 22.0);
    }
}

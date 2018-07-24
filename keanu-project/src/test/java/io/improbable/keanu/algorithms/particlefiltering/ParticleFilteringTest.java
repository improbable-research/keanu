package io.improbable.keanu.algorithms.particlefiltering;

import static java.lang.Math.exp;

import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class ParticleFilteringTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    private final Logger log = LoggerFactory.getLogger(ParticleFilteringTest.class);

    @Test
    public void findsCorrectTemp() {

        DoubleVertex temperature = VertexOfType.uniform(0.0, 100.0);
        DoubleVertex noiseAMu = VertexOfType.gaussian(0.0, 2.0);
        DoubleVertex noiseA = VertexOfType.gaussian(noiseAMu, ConstantVertex.of(2.0));
        DoubleVertex noiseBMu = VertexOfType.gaussian(0.0, 2.0);
        DoubleVertex noiseB = VertexOfType.gaussian(noiseBMu, ConstantVertex.of(2.0));
        DoubleVertex noiseCMu = VertexOfType.gaussian(0.0, 2.0);
        DoubleVertex noiseC = VertexOfType.gaussian(noiseCMu, ConstantVertex.of(2.0));
        DoubleVertex noiseDMu = VertexOfType.gaussian(0.0, 2.0);
        DoubleVertex noiseD = VertexOfType.gaussian(noiseDMu, ConstantVertex.of(2.0));
        DoubleVertex thermometerA = VertexOfType.gaussian(temperature.plus(noiseA), ConstantVertex.of(1.0));
        DoubleVertex thermometerB = VertexOfType.gaussian(temperature.plus(noiseB), ConstantVertex.of(1.0));
        DoubleVertex thermometerC = VertexOfType.gaussian(temperature.plus(noiseC), ConstantVertex.of(1.0));
        DoubleVertex thermometerD = VertexOfType.gaussian(temperature.plus(noiseD), ConstantVertex.of(1.0));
        thermometerA.observe(DoubleTensor.scalar(21.0));
        thermometerB.observe(DoubleTensor.scalar(19.5));
        thermometerC.observe(DoubleTensor.scalar(22.0));
        thermometerD.observe(DoubleTensor.scalar(18.0));

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

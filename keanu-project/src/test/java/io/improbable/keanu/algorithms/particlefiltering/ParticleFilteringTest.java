package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorUniformVertex;
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

        DoubleTensorVertex temperature = new TensorUniformVertex(0.0, 100.0);
        DoubleTensorVertex noiseAMu = new TensorGaussianVertex(0.0, 2.0);
        DoubleTensorVertex noiseA = new TensorGaussianVertex(noiseAMu, 2.0);
        DoubleTensorVertex noiseBMu = new TensorGaussianVertex(0.0, 2.0);
        DoubleTensorVertex noiseB = new TensorGaussianVertex(noiseBMu, 2.0);
        DoubleTensorVertex noiseCMu = new TensorGaussianVertex(0.0, 2.0);
        DoubleTensorVertex noiseC = new TensorGaussianVertex(noiseCMu, 2.0);
        DoubleTensorVertex noiseDMu = new TensorGaussianVertex(0.0, 2.0);
        DoubleTensorVertex noiseD = new TensorGaussianVertex(noiseDMu, 2.0);
        DoubleTensorVertex thermometerA = new TensorGaussianVertex(temperature.plus(noiseA), 1.0);
        DoubleTensorVertex thermometerB = new TensorGaussianVertex(temperature.plus(noiseB), 1.0);
        DoubleTensorVertex thermometerC = new TensorGaussianVertex(temperature.plus(noiseC), 1.0);
        DoubleTensorVertex thermometerD = new TensorGaussianVertex(temperature.plus(noiseD), 1.0);
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

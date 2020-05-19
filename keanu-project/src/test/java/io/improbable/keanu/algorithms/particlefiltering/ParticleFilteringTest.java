package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import lombok.extern.slf4j.Slf4j;
import org.junit.Rule;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

@Slf4j
public class ParticleFilteringTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

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
        ParticleFilter particleFilter = ParticleFilter.ofVertexInGraph(temperature)
            .withNumParticles(numParticles)
            .withResamplingCycles(resamplingCycles)
            .withResamplingProportion(resamplingProportion)
            .build();

        Particle mostProbableParticle = particleFilter.getMostProbableParticle();

        double estimatedTemp = mostProbableParticle.getScalarValueOfVertex(temperature);
        double probability = mostProbableParticle.logProb();

        log.info("Final temp estimate = " + estimatedTemp + ", log probability = " + probability);

        assertTrue(estimatedTemp > 18.0);
        assertTrue(estimatedTemp < 22.0);
    }
}

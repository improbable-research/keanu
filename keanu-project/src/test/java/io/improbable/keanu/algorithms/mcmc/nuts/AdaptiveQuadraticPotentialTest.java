package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.assertEquals;

public class AdaptiveQuadraticPotentialTest {


    @Test
    public void doesQuadraticPotentialBeforeAnyUpdate() {
        DoubleVertex v = new GaussianVertex(0, 1);
        v.setValue(1);

        Map<VariableReference, DoubleTensor> position = ImmutableMap.of(v.getReference(), v.getValue());

        AdaptiveQuadraticPotential potential = new AdaptiveQuadraticPotential(
            0, 1,
            10,
            100
        );

        potential.initialize(position);

        Map<VariableReference, DoubleTensor> momentum = ImmutableMap.of(v.getReference(), DoubleTensor.scalar(0.5));
        Map<VariableReference, DoubleTensor> velocity = potential.getVelocity(momentum);

        double kineticEnergy = potential.getKineticEnergy(momentum, velocity);

        assertEquals(0.5, velocity.get(v.getReference()).scalar(), 1e-6);
        assertEquals(0.5 * Math.pow(0.5, 2), kineticEnergy, 1e-6);
    }

    /**
     * The random momentum variance should be inversely proportional to the sample variance
     */
    @Test
    public void doesUpdateAfterAdaptSample() {
        DoubleVertex v = new GaussianVertex(0, 1);
        v.setValue(1.0);

        Map<VariableReference, DoubleTensor> position = ImmutableMap.of(v.getReference(), v.getValue());

        KeanuRandom random = new KeanuRandom(0);

        AdaptiveQuadraticPotential potential = new AdaptiveQuadraticPotential(
            0, 1,
            0,
            1500
        );

        potential.initialize(position);

        double targetStandardDeviation = 2;
        for (int i = 0; i < 1000; i++) {
            double r = random.nextGaussian() * targetStandardDeviation;
            potential.update(ImmutableMap.of(v.getReference(), DoubleTensor.scalar(r)));
        }

        SummaryStatistics statistics = new SummaryStatistics();
        for (int i = 0; i < 1000; i++) {
            double sample = potential.randomMomentum(random).get(v.getReference()).scalar();
            statistics.addValue(sample);
        }

        assertEquals(1.0 / targetStandardDeviation, statistics.getStandardDeviation(), 1e-2);
    }

    /**
     * Samples from 2 windows ago should not affect the potential variance
     */
    @Test
    public void doesUseWindowsForAdaption() {
        DoubleVertex v = new GaussianVertex(0, 1);
        v.setValue(1.0);

        Map<VariableReference, DoubleTensor> position = ImmutableMap.of(v.getReference(), v.getValue());

        KeanuRandom random = new KeanuRandom(0);

        int windowSize = 1000;
        AdaptiveQuadraticPotential potential = new AdaptiveQuadraticPotential(
            0, 1,
            0,
            windowSize
        );

        potential.initialize(position);

        SummaryStatistics statisticsWindow1And2 = new SummaryStatistics();

        double targetStandardDeviationWindow1 = 2;
        for (int i = 0; i < windowSize; i++) {
            double r = random.nextGaussian() * targetStandardDeviationWindow1;
            statisticsWindow1And2.addValue(r);
            potential.update(ImmutableMap.of(v.getReference(), DoubleTensor.scalar(r)));
        }

        assertEquals(
            statisticsWindow1And2.getStandardDeviation(),
            potential.getStandardDeviation().get(v.getReference()).scalar(), 1e-2
        );

        SummaryStatistics statisticsWindow2And3 = new SummaryStatistics();

        double targetStandardDeviationWindow2 = 3;
        for (int i = 0; i < windowSize; i++) {
            double r = random.nextGaussian() * targetStandardDeviationWindow2;
            statisticsWindow1And2.addValue(r);
            statisticsWindow2And3.addValue(r);
            potential.update(ImmutableMap.of(v.getReference(), DoubleTensor.scalar(r)));
        }

        assertEquals(
            statisticsWindow1And2.getStandardDeviation(),
            potential.getStandardDeviation().get(v.getReference()).scalar(),
            1e-2
        );

        SummaryStatistics statisticsWindow3And4 = new SummaryStatistics();

        double targetStandardDeviationWindow3 = 4;
        for (int i = 0; i < windowSize; i++) {
            double r = random.nextGaussian() * targetStandardDeviationWindow3;
            statisticsWindow2And3.addValue(r);
            statisticsWindow3And4.addValue(r);
            potential.update(ImmutableMap.of(v.getReference(), DoubleTensor.scalar(r)));
        }

        assertEquals(
            statisticsWindow2And3.getStandardDeviation(),
            potential.getStandardDeviation().get(v.getReference()).scalar(),
            1e-2
        );

        potential.update(ImmutableMap.of(v.getReference(), DoubleTensor.scalar(random.nextGaussian() * targetStandardDeviationWindow3)));

        assertEquals(
            statisticsWindow3And4.getStandardDeviation(),
            potential.getStandardDeviation().get(v.getReference()).scalar(),
            1e-2
        );
    }
}

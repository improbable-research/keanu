package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;
import java.util.Map;

import static io.improbable.keanu.tensor.dbl.DoubleTensor.scalar;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;
import static org.junit.Assert.assertEquals;

public class LeapfrogIntegratorTest {

    private GaussianVertex vertex;
    private KeanuProbabilisticModelWithGradient gradientCalculator;
    private LeapfrogState start;
    private LeapfrogIntegrator integrator;

    private double initialPosition = 0.0;
    private double initialMomentum = 0.5;

    @Before
    public void setup() {
        vertex = new GaussianVertex(0, 1);

        gradientCalculator = new KeanuProbabilisticModelWithGradient(
            new BayesianNetwork(vertex.getConnectedGraph())
        );

        Map<VariableReference, DoubleTensor> p = ImmutableMap.of(vertex.getId(), scalar(0));

        Potential potential = new AdaptiveQuadraticPotential(0, 1, 1, 101);

        potential.initialize(p);

        start = leapfrogAt(vertex, initialPosition, initialMomentum, potential);

        integrator = new LeapfrogIntegrator(potential);
    }

    @Test
    public void canLeapForward() {

        LeapfrogState leap = integrator.step(start, gradientCalculator, 1.0);
        assertEquals(0.5, leap.getPosition().get(vertex.getId()).scalar(), 1e-6);
    }

    @Test
    public void canLeapForwardAndBack() {

        LeapfrogState leap = integrator.step(start, gradientCalculator, 1.0);
        assertEquals(initialMomentum, leap.getPosition().get(vertex.getId()).scalar(), 1e-6);

        LeapfrogState leapBack = integrator.step(leap, gradientCalculator, -1.0);
        assertEquals(initialPosition, leapBack.getPosition().get(vertex.getId()).scalar(), 1e-6);
    }

    @Test
    public void leapsForwardWithMinimalEnergyLoss() {

        LeapfrogState leap = start;
        for (int i = 0; i < 1000; i++) {
            leap = integrator.step(leap, gradientCalculator, 1e-3);
        }

        assertEquals(start.getEnergy(), leap.getEnergy(), 1e-6);
    }

    @Test
    public void doesDecreaseKineticEnergyWhenLogProbDecreases() {

        double startEnergy = start.getKineticEnergy();

        LeapfrogState leap = integrator.step(start, gradientCalculator, 1e-3);

        double afterLeapEnergy = leap.getKineticEnergy();

        assertThat(startEnergy, greaterThan(afterLeapEnergy));
    }

    public static LeapfrogState leapfrogAt(GaussianVertex vertex, double position, double momentum, Potential potential) {

        DoubleTensor tensorPosition = scalar(position);

        double gradient = vertex.dLogProb(tensorPosition, Collections.singleton(vertex)).get(vertex).scalar();
        double logProb = vertex.logProb(tensorPosition);

        Map<VariableReference, DoubleTensor> p = ImmutableMap.of(vertex.getId(), tensorPosition);
        Map<VariableReference, DoubleTensor> m = ImmutableMap.of(vertex.getId(), scalar(momentum));
        Map<VariableReference, DoubleTensor> g = ImmutableMap.of(vertex.getId(), scalar(gradient));

        return new LeapfrogState(p, m, g, logProb, potential);
    }

}
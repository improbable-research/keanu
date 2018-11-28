package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class GradientOptimizerTest {

    @Test
    public void doesCallOnFitnessAndOnGradientHandler() {
        AtomicInteger fitnessTimesCalled = new AtomicInteger(0);
        AtomicInteger gradientTimesCalled = new AtomicInteger(0);
        GradientOptimizer optimizer = GradientOptimizer.ofConnectedGraph(
            new GaussianVertex(0, 1)
        );
        optimizer.addFitnessCalculationHandler((point, fitness) -> fitnessTimesCalled.incrementAndGet());
        optimizer.addGradientCalculationHandler((point, fitness) -> gradientTimesCalled.incrementAndGet());
        optimizer.maxAPosteriori();

        assertTrue(fitnessTimesCalled.get() > 0);
        assertTrue(gradientTimesCalled.get() > 0);
    }

    @Test(expected = UnsupportedOperationException.class)
    public void errorOnDiscreteLatents() {
        PoissonVertex v1 = new PoissonVertex(15);
        PoissonVertex v2 = new PoissonVertex(v1);

        GradientOptimizer optimizer = GradientOptimizer.ofConnectedGraph(v1);
    }

    @Test
    public void shouldAllowObservationChange() {

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(A.abs(), 1);
        A.observe(2.0);

        GradientOptimizer optimizer = GradientOptimizer.ofConnectedGraph(B);
        optimizer.maxAPosteriori();

        assertEquals(2.0, B.getValue().scalar(), 1e-5);

        A.observe(3.0);

        GradientOptimizer optimizerAfterObserve = GradientOptimizer.ofConnectedGraph(B);
        optimizerAfterObserve.maxAPosteriori();

        assertEquals(3.0, B.getValue().scalar(), 1e-5);

    }
}

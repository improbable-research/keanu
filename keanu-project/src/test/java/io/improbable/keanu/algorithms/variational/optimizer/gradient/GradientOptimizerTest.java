package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import static org.junit.Assert.assertTrue;

import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;

public class GradientOptimizerTest {

    @Test
    public void doesCallOnFitnessAndOnGradientHandler() {
        AtomicInteger fitnessTimesCalled = new AtomicInteger(0);
        AtomicInteger gradientTimesCalled = new AtomicInteger(0);
        GradientOptimizer optimizer = GradientOptimizer.ofConnectedGraph(new GaussianVertex(0, 1));
        optimizer.addFitnessCalculationHandler(
                (point, fitness) -> fitnessTimesCalled.incrementAndGet());
        optimizer.addGradientCalculationHandler(
                (point, fitness) -> gradientTimesCalled.incrementAndGet());
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
}

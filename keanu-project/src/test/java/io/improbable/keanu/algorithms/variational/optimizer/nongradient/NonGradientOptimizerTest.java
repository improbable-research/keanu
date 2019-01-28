package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.variational.optimizer.KeanuOptimizer;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertTrue;

public class NonGradientOptimizerTest {

    @Test
    public void doesCallOnFitnessHandler() {
        AtomicInteger timesCalled = new AtomicInteger(0);
        NonGradientOptimizer optimizer = KeanuOptimizer.NonGradient.ofConnectedGraph(new GaussianVertex(new GaussianVertex(0, 1), 1));
        optimizer.addFitnessCalculationHandler((point, fitness) -> timesCalled.incrementAndGet());
        optimizer.maxAPosteriori();
        assertTrue(timesCalled.get() > 0);
    }

}

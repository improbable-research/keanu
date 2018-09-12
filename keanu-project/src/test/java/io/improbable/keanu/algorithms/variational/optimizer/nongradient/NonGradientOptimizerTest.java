package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

public class NonGradientOptimizerTest {

    @Test
    public void doesCallOnFitnessHandler() {
        AtomicInteger timesCalled = new AtomicInteger(0);
        NonGradientOptimizer optimizer = NonGradientOptimizer.ofConnectedGraph(new GaussianVertex(new GaussianVertex(0, 1), 1));
        optimizer.addFitnessCalculationHandler((point, fitness) -> timesCalled.incrementAndGet());
        optimizer.maxAPosteriori();
        assertTrue(timesCalled.get() > 0);
    }

    @Test
    public void canFindMAPGivenBounds() {

        DoubleVertex A = new GaussianVertex(new int[]{1, 2}, ConstantVertex.of(new double[]{1, -3}), 1);
        A.setValue(new double[]{0, 0});

        OptimizerBounds bounds = new OptimizerBounds();
        bounds.addBound(A, DoubleTensor.create(new double[]{-1, -2}), 0.9);

        NonGradientOptimizer optimizer = NonGradientOptimizer.builder()
            .boundsRange(10)
            .optimizerBounds(bounds)
            .bayesianNetwork(new BayesianNetwork(A.getConnectedGraph()))
            .build();

        optimizer.maxAPosteriori();

        assertArrayEquals(new double[]{0.9, -2}, A.getValue().asFlatDoubleArray(), 1e-2);
    }

}

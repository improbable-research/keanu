package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class NonGradientOptimizerTest {

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

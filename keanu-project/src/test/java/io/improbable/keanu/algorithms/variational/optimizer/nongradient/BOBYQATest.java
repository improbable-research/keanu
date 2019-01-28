package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.variational.optimizer.KeanuOptimizer;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class BOBYQATest {

    @Test
    public void canFindMAPGivenBounds() {

        DoubleVertex A = new GaussianVertex(new long[]{2}, ConstantVertex.of(new double[]{1, -3}), 1);
        A.setValue(new double[]{0, 0});

        OptimizerBounds bounds = new OptimizerBounds();
        bounds.addBound(A.getId(), DoubleTensor.create(-1, -2), 0.9);

        NonGradientOptimizer optimizer = KeanuOptimizer.NonGradient
            .builderFor(A.getConnectedGraph())
            .algorithm(BOBYQA.builder()
                .boundsRange(10)
                .optimizerBounds(bounds)
                .build())
            .build();

        optimizer.maxAPosteriori();

        assertArrayEquals(new double[]{0.9, -2}, A.getValue().asFlatDoubleArray(), 1e-2);
    }
}

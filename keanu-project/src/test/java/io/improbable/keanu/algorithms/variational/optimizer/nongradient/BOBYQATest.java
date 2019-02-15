package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.Keanu;
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

        NonGradientOptimizer optimizer = Keanu.Optimizer.NonGradient
            .builderFor(A.getConnectedGraph())
            .algorithm(BOBYQA.builder()
                .boundsRange(10)
                .optimizerBounds(bounds)
                .build())
            .build();

        optimizer.maxAPosteriori();

        assertArrayEquals(new double[]{0.9, -2}, A.getValue().asFlatDoubleArray(), 1e-2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsIfLessThanTwoDimensions() {
        DoubleVertex A = new GaussianVertex(0, 1);

        OptimizerBounds bounds = new OptimizerBounds();
        bounds.addBound(A.getId(), -1.0, 1.);

        NonGradientOptimizer optimizer = Keanu.Optimizer.NonGradient.builderFor(A.getConnectedGraph()).algorithm(BOBYQA.builder().boundsRange(10).optimizerBounds(bounds).build()).build();
        optimizer.maxAPosteriori();
    }
}

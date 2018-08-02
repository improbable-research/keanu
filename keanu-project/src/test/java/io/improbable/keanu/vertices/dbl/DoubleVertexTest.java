package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class DoubleVertexTest {

    @Test
    public void canObserveArrayOfValues() {
        DoubleVertex gaussianVertex = new GaussianVertex(0, 1);
        double[] observation = new double[]{1, 2, 3};
        gaussianVertex.observe(observation);
        assertArrayEquals(observation, gaussianVertex.getValue().asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetAndCascadeArrayOfValues() {
        DoubleVertex gaussianVertex = new GaussianVertex(0, 1);
        double[] values = new double[]{1, 2, 3};
        gaussianVertex.setAndCascade(values);
        assertArrayEquals(values, gaussianVertex.getValue().asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetValueArrayOfValues() {
        DoubleVertex gaussianVertex = new GaussianVertex(0, 1);
        double[] values = new double[]{1, 2, 3};
        gaussianVertex.setValue(values);
        assertArrayEquals(values, gaussianVertex.getValue().asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetValueAsScalarOnNonScalarVertex() {
        DoubleVertex gaussianVertex = new GaussianVertex(new int[]{1, 2}, 0, 1);
        gaussianVertex.setValue(2);
        assertArrayEquals(new double[]{2}, gaussianVertex.getValue().asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetAndCascadeAsScalarOnNonScalarVertex() {
        DoubleVertex gaussianVertex = new GaussianVertex(new int[]{1, 2}, 0, 1);
        gaussianVertex.setAndCascade(2);
        assertArrayEquals(new double[]{2}, gaussianVertex.getValue().asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canPluckValue() {
        DoubleVertex gaussianVertex = new GaussianVertex(0, 1);
        double[] values = new double[]{1, 2, 3};
        gaussianVertex.setAndCascade(values);
        assertEquals(1, gaussianVertex.pluck(0, 0).getValue().scalar(), 0.0);
    }
}

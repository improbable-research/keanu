package io.improbable.keanu.vertices.dbl;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;

import org.junit.Test;

import io.improbable.keanu.tensor.TensorMatchers;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class DoubleVertexTest {

    @Test
    public void canObserveArrayOfValues() {
        DoubleVertex gaussianVertex = new GaussianVertex(0, 1);
        double[] observation = new double[]{1, 2, 3};
        gaussianVertex.observe(observation);
        assertArrayEquals(observation, gaussianVertex.getValue().asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canObserveTensor() {
        DoubleVertex gaussianVertex = new GaussianVertex(0, 1);
        DoubleTensor observation = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        gaussianVertex.observe(observation);
        assertArrayEquals(observation.asFlatDoubleArray(), gaussianVertex.getValue().asFlatDoubleArray(), 0.0);
        assertArrayEquals(observation.getShape(), gaussianVertex.getShape());
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
    public void canTakeValue() {
        DoubleVertex gaussianVertex = new GaussianVertex(0, 1);
        double[] values = new double[]{1, 2, 3};
        gaussianVertex.setAndCascade(values);
        assertEquals(1, gaussianVertex.take(0, 0).getValue().scalar(), 0.0);
    }

    @Test
    public void canReshape() {
        DoubleVertex gaussianVertex = new GaussianVertex(0, 1);
        gaussianVertex.setAndCascade(DoubleTensor.ones(2, 2));
        assertArrayEquals(gaussianVertex.getShape(), new int[]{2, 2});
        DoubleVertex reshaped = gaussianVertex.reshape(4, 1);
        assertArrayEquals(reshaped.getShape(), new int[]{4, 1});
    }

    @Test
    public void canConcat() {
        DoubleVertex A = new UniformVertex(0, 1);
        A.setValue(DoubleTensor.arange(1, 5).reshape(2, 2));

        DoubleVertex B = new UniformVertex(0, 1);
        B.setValue(DoubleTensor.arange(5, 9).reshape(2, 2));

        DoubleVertex concatDimZero = DoubleVertex.concat(0, A, B);
        assertArrayEquals(concatDimZero.getShape(), new int[]{4, 2});

        DoubleVertex concatDimOne = DoubleVertex.concat(1, A, B);
        assertArrayEquals(concatDimOne.getShape(), new int[]{2, 4});
    }

    @Test
    public void sampleScalarsAsTensorFillsGivenShapeWithSamples() {
        DoubleVertex vertex = new DoubleVertex() {
            double sampledValue = 0;

            @Override
            public DoubleTensor sample(KeanuRandom random) {
                sampledValue += 1;
                return DoubleTensor.scalar(sampledValue);
            }
        };

        final int[] shape = new int[]{2, 2, 2};
        final DoubleTensor expected = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8},
            shape);
        final DoubleTensor actual = vertex.sampleScalarValuesAsTensor(shape);

        assertThat(actual, TensorMatchers.isEqualTo(expected));
    }

    @Test(expected = IllegalArgumentException.class)
    public void sampleScalarsAsTensorFailsForNonScalars() {
        DoubleVertex vertex = new GaussianVertex(new int[]{2, 2}, 0, 1);

        vertex.sampleScalarValuesAsTensor(new int[]{2, 2});
    }

}

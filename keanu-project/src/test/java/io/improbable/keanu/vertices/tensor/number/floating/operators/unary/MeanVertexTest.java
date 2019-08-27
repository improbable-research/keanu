package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers;
import io.improbable.keanu.vertices.tensor.number.floating.FloatingPointTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.finiteDifferenceMatchesElementwise;
import static org.junit.Assert.assertEquals;

public class MeanVertexTest {

    @Test
    public void doesOperateOnMatrix() {
        UnaryOperationTestHelpers.operatesOnInput(FloatingPointTensor::mean, FloatingPointTensorVertex::mean);
    }

    @Test
    public void doesMeanAllDimensions() {
        DoubleVertex a = new UniformVertex(new long[]{1, 5}, 0, 10);
        a.setValue(new double[]{1, 2, 3, 4, 5});

        DoubleVertex meaned = a.mean();

        assertEquals((1 + 2 + 3 + 4 + 5) / 5.0, meaned.eval().scalar(), 1e-5);
    }

    @Test
    public void doesMeanAllSpecifiedDimensions() {
        DoubleVertex a = new UniformVertex(new long[]{1, 5}, 0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4, 5}, 1, 5));

        DoubleVertex meaned = a.mean(0, 1);
        DoubleTensor expected = DoubleTensor.scalar(1 + 2 + 3 + 4 + 5).div(5);

        assertEquals(expected, meaned.eval());
    }

    @Test
    public void doesMeanSingleSpecifiedDimensions() {
        DoubleVertex a = new UniformVertex(new long[]{2, 3}, 0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6}, 2, 3));

        DoubleVertex meaned = a.mean(1);
        DoubleTensor expected = DoubleTensor.create(6, 15).div(3);

        assertEquals(expected, meaned.eval());
    }

    @Test
    public void changesMatchGradient() {
        finiteDifferenceMatchesElementwise(DoubleVertex::mean);
    }

    @Test
    public void changesMatchGradientWhenMeanDim0() {
        UniformVertex inputVertex = new UniformVertex(new long[]{2}, -10.0, 10.0);
        inputVertex.setValue(DoubleTensor.arange(1, 3));

        DoubleVertex outputVertex = inputVertex.mean(0);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 1e-3, 1e-4);
    }

    @Test
    public void changesMatchGradientWhenMeanSpecificDimensions() {
        UniformVertex inputVertex = new UniformVertex(new long[]{2, 2, 2}, -10.0, 10.0);
        inputVertex.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));

        DoubleVertex outputVertex = inputVertex.mean(0)
            .times(
                inputVertex.mean(1)
            ).times(
                inputVertex.mean(2)
            ).times(
                inputVertex.mean()
            );

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 1e-3, 1e-4);
    }

}

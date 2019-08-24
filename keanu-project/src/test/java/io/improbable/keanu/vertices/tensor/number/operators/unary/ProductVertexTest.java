package io.improbable.keanu.vertices.tensor.number.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Ignore;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;

public class ProductVertexTest {

    @Test
    public void changesMatchGradientProductAll() {
        UniformVertex inputA = new UniformVertex(new long[]{3}, -10.0, 10.0);
        DoubleVertex product = inputA.product();
        DoubleVertex outputVertex = product.times(
            new ConstantDoubleVertex(DoubleTensor.arange(2))
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }

    @Test
    public void changesMatchGradientProductDim1() {
        UniformVertex inputA = new UniformVertex(new long[]{2, 3}, -10.0, 10.0);
        DoubleVertex product = inputA.product(1);
        DoubleVertex outputVertex = product.times(
            new ConstantDoubleVertex(DoubleTensor.arange(2))
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }

    @Test
    @Ignore
    public void changesMatchGradientWhenProductDimensionsFromRight() {
        UniformVertex inputVertex = new UniformVertex(new long[]{2, 2, 2}, -10.0, 10.0);
        inputVertex.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));

        DoubleVertex outputVertex = inputVertex.product(-1)
            .times(
                inputVertex.product(-2)
            ).times(
                inputVertex.product(-3)
            ).times(
                inputVertex.product()
            );

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 1e-6, 1e-6);
    }

    @Test
    public void changesMatchGradientProductDim0() {
        UniformVertex inputA = new UniformVertex(new long[]{2, 3}, -10.0, 10.0);
        DoubleVertex product = inputA.product(0);
        DoubleVertex outputVertex = product.times(
            new ConstantDoubleVertex(DoubleTensor.arange(3))
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }
}

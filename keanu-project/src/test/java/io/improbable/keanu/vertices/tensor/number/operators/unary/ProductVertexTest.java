package io.improbable.keanu.vertices.tensor.number.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Ignore;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;

public class ProductVertexTest {

    @Test
    @Ignore
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
    @Ignore
    public void changesMatchGradientProductDim1() {
        UniformVertex inputA = new UniformVertex(new long[]{3, 3}, -10.0, 10.0);
        DoubleVertex product = inputA.product(1);
        DoubleVertex outputVertex = product.times(
            new ConstantDoubleVertex(DoubleTensor.arange(2))
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }
}

package io.improbable.keanu.vertices.tensor.number.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class AbsVertexTest {

    @Test
    public void absTwoScalarVertexValues() {
        operatesOnScalarVertexValue(
            -3,
            3.0,
            DoubleVertex::abs
        );
    }

    @Test
    public void absTwoMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{-2, 3.0, -6.0, 4.0},
            new double[]{2.0, 3.0, 6.0, 4.0},
            DoubleVertex::abs
        );
    }

    @Test
    public void changesMatchGradient() {
        UniformVertex inputVertex = new UniformVertex(new long[]{2, 2, 2}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex.abs();

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 1e-6, 1e-10);
    }

    @Test
    public void changesMatchGradientWithInsulatedOp() {
        UniformVertex inputVertex = new UniformVertex(new long[]{2, 2, 1}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex
            .times(inputVertex.sum(2))
            .abs()
            .sum(1);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 1e-6, 1e-10);
    }
}

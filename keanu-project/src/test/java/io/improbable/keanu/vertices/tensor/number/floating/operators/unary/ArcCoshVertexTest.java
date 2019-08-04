package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.apache.commons.math3.util.FastMath;
import org.junit.Test;

import java.util.function.Function;

import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class ArcCoshVertexTest {

    @Test
    public void acoshScalarVertexValue() {
        operatesOnScalarVertexValue(
            Math.PI,
            FastMath.acosh(Math.PI),
            DoubleVertex::acosh
        );
    }

    @Test
    public void calculatesDerivativeOScalarACosh() {
        calculatesDerivativeOfScalar(
            1.5,
            1.0 / Math.sqrt(1.5 * 1.5 -1),
            DoubleVertex::acosh
        );
    }

    @Test
    public void acoshMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{FastMath.acosh(0.0), FastMath.acosh(0.1), FastMath.acosh(0.2), FastMath.acosh(0.3)},
            DoubleVertex::acosh
        );
    }

    @Test
    public void calculatesDerivativeOfMatrixElementWiseACosh() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{1.1, 1.2, 1.3, 1.4},
            toDiagonalArray(new double[]{
                1.0 / Math.sqrt(1.1 * 1.1 - 1),
                1.0 / Math.sqrt(1.2 * 1.2 - 1),
                1.0 / Math.sqrt(1.3 * 1.3 - 1),
                1.0 / Math.sqrt(1.4 * 1.4 - 1)
            }),
            DoubleVertex::acosh
        );
    }

    @Test
    public void changesMatchGradient() {
        finiteDifferenceMatchesElementwise(DoubleVertex::acosh);
    }

    public <T extends DoubleVertex> void finiteDifferenceMatchesElementwise(Function<UniformVertex, T> op) {
        testWithFiniteDifference(op, new long[0]);
        testWithFiniteDifference(op, new long[]{3});
        testWithFiniteDifference(op, new long[]{2, 3});
        testWithFiniteDifference(op, new long[]{2, 2, 2});
    }

    public <T extends DoubleVertex> void testWithFiniteDifference(Function<UniformVertex, T> op, long[] shape) {
        UniformVertex A = new UniformVertex(shape, 1.1, 1.9);
        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(A), op.apply(A), 1e-10, 1e-10);
    }

}

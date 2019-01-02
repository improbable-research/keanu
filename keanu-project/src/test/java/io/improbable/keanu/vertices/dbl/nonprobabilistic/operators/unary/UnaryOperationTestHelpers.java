package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class UnaryOperationTestHelpers {

    public static void operatesOnScalarVertexValue(double aValue,
                                                   double expected,
                                                   Function<DoubleVertex, DoubleVertex> op) {

        ConstantDoubleVertex A = ConstantVertex.of(aValue);

        assertEquals(expected, op.apply(A).getValue().scalar(), 1e-5);
    }

    public static <T extends DoubleVertex & Differentiable> void calculatesDerivativeOfScalar(double aValue,
                                                    double expectedGradientWrtA,
                                                    Function<DoubleVertex, T> op) {

        UniformVertex A = new UniformVertex(0.0, 1.0);
        A.setAndCascade(Nd4jDoubleTensor.scalar(aValue));
        T output = op.apply(A);

        DoubleTensor wrtAForward = Differentiator.forwardModeAutoDiff(A, output).of(output).get();
        assertEquals(
            expectedGradientWrtA,
            wrtAForward.scalar(),
            1e-5
        );

        DoubleTensor wrtAReverse = Differentiator.reverseModeAutoDiff(output, A).withRespectTo(A);
        assertEquals(
            expectedGradientWrtA,
            wrtAReverse.scalar(),
            1e-5
        );
    }

    public static void operatesOn2x2MatrixVertexValues(double[] aValues,
                                                       double[] expected,
                                                       Function<DoubleVertex, DoubleVertex> op) {

        ConstantDoubleVertex A = new ConstantDoubleVertex(DoubleTensor.create(aValues, new long[]{2, 2}));

        DoubleTensor result = op.apply(A).getValue();

        DoubleTensor expectedTensor = Nd4jDoubleTensor.create(expected, new long[]{2, 2});

        assertEquals(expectedTensor.getValue(0, 0), result.getValue(0, 0), 1e-5);
        assertEquals(expectedTensor.getValue(0, 1), result.getValue(0, 1), 1e-5);
        assertEquals(expectedTensor.getValue(1, 0), result.getValue(1, 0), 1e-5);
        assertEquals(expectedTensor.getValue(1, 1), result.getValue(1, 1), 1e-5);
    }

    public static <T extends DoubleVertex & Differentiable> void calculatesDerivativeOfMatrixElementWiseOperator(double[] aValues,
                                                                       double[] expectedGradientWrtA,
                                                                       Function<DoubleVertex, T> op) {

        long[] matrixShape = new long[]{2, 2};
        long[] expectedShape = TensorShape.concat(matrixShape, matrixShape);
        UniformVertex A = new UniformVertex(matrixShape, 0.0, 1.0);
        A.setAndCascade(Nd4jDoubleTensor.create(aValues, matrixShape));

        T output = op.apply(A);

        PartialDerivative result = Differentiator.forwardModeAutoDiff(A, output).of(output);
        DoubleTensor wrtAForward = result.get();
        assertArrayEquals(expectedGradientWrtA, wrtAForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedShape, wrtAForward.getShape());

        DoubleTensor wrtAReverse = Differentiator.reverseModeAutoDiff(output, A).withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA, wrtAReverse.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedShape, wrtAReverse.getShape());
    }

}

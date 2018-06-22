package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import java.util.function.BiFunction;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class BinaryOperationTestHelpers {

    public static void operatesOnTwoScalarVertexValues(double aValue,
                                                       double bValue,
                                                       double expected,
                                                       BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {

        UniformVertex A = new UniformVertex(0.0, 1.0);
        A.setAndCascade(Nd4jDoubleTensor.scalar(aValue));
        UniformVertex B = new UniformVertex(0.0, 1.0);
        B.setAndCascade(Nd4jDoubleTensor.scalar(bValue));

        assertEquals(expected, op.apply(A, B).getValue().scalar(), 1e-5);
    }

    public static void calculatesDualNumberOfTwoScalars(double aValue,
                                                        double bValue,
                                                        double expectedGradientWrtA,
                                                        double expectedGradientWrtB,
                                                        BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {

        UniformVertex A = new UniformVertex(0.0, 1.0);
        A.setAndCascade(Nd4jDoubleTensor.scalar(aValue));
        UniformVertex B = new UniformVertex(0.0, 1.0);
        B.setAndCascade(Nd4jDoubleTensor.scalar(bValue));

        DualNumber resultDualNumber = op.apply(A, B).getDualNumber();
        assertEquals(expectedGradientWrtA, resultDualNumber.getPartialDerivatives().withRespectTo(A).scalar(), 1e-5);
        assertEquals(expectedGradientWrtB, resultDualNumber.getPartialDerivatives().withRespectTo(B).scalar(), 1e-5);
    }

    public static void operatesOnTwo2x2MatrixVertexValues(double[] aValues,
                                                          double[] bValues,
                                                          double[] expected,
                                                          BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {

        UniformVertex A = new UniformVertex(new int[]{2, 2}, 0.0, 1.0);
        A.setAndCascade(Nd4jDoubleTensor.create(aValues, new int[]{2, 2}));
        UniformVertex B = new UniformVertex(new int[]{2, 2}, 0.0, 1.0);
        B.setAndCascade(Nd4jDoubleTensor.create(bValues, new int[]{2, 2}));

        DoubleTensor result = op.apply(A, B).getValue();

        DoubleTensor expectedTensor = Nd4jDoubleTensor.create(expected, new int[]{2, 2});

        assertEquals(expectedTensor.getValue(0, 0), result.getValue(0, 0), 1e-5);
        assertEquals(expectedTensor.getValue(0, 1), result.getValue(0, 1), 1e-5);
        assertEquals(expectedTensor.getValue(1, 0), result.getValue(1, 0), 1e-5);
        assertEquals(expectedTensor.getValue(1, 1), result.getValue(1, 1), 1e-5);
    }

    public static void calculatesDualNumberOfTwoMatricesElementWiseOperator(double[] aValues,
                                                                            double[] bValues,
                                                                            double[] expectedGradientWrtA,
                                                                            double[] expectedGradientWrtB,
                                                                            BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {

        int[] matrixShape = new int[]{2, 2};
        UniformVertex A = new UniformVertex(matrixShape, 0.0, 1.0);
        A.setAndCascade(Nd4jDoubleTensor.create(aValues, matrixShape));
        UniformVertex B = new UniformVertex(matrixShape, 0.0, 1.0);
        B.setAndCascade(Nd4jDoubleTensor.create(bValues, matrixShape));

        DualNumber result = op.apply(A, B).getDualNumber();

        DoubleTensor wrtA = result.getPartialDerivatives().withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA, wrtA.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(TensorShape.concat(matrixShape, matrixShape), wrtA.getShape());

        DoubleTensor wrtB = result.getPartialDerivatives().withRespectTo(B);
        assertArrayEquals(expectedGradientWrtB, wrtB.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(TensorShape.concat(matrixShape, matrixShape), wrtB.getShape());
    }

    public static void calculatesDualNumberOfTwoVectorsElementWiseOperator(double[] aValues,
                                                                           double[] bValues,
                                                                           double[] expectedGradientWrtA,
                                                                           double[] expectedGradientWrtB,
                                                                           BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        UniformVertex A = new UniformVertex(new int[]{1, aValues.length}, 0.0, 1.0);
        A.setAndCascade(Nd4jDoubleTensor.create(aValues, new int[]{1, aValues.length}));
        UniformVertex B = new UniformVertex(new int[]{1, bValues.length}, 0.0, 1.0);
        B.setAndCascade(Nd4jDoubleTensor.create(bValues, new int[]{1, bValues.length}));

        DualNumber result = op.apply(A, B).getDualNumber();
        DoubleTensor expectedTensorA = Nd4jDoubleTensor.create(expectedGradientWrtA, new int[]{1, aValues.length, 1, aValues.length});

        DoubleTensor wrtA = result.getPartialDerivatives().withRespectTo(A);
        for (int i = 0; i < aValues.length; i++) {
            assertEquals(expectedTensorA.getValue(0, i, 0, i), wrtA.getValue(0, i, 0, i), 1e-10);
        }

        DoubleTensor expectedTensorB = Nd4jDoubleTensor.create(expectedGradientWrtB, new int[]{2, 2, 2, 2});
        DoubleTensor wrtB = result.getPartialDerivatives().withRespectTo(B);
        for (int i = 0; i < aValues.length; i++) {
            assertEquals(expectedTensorB.getValue(0, i, 0, i), wrtB.getValue(0, i, 0, i), 1e-10);
        }
    }

    public static void calculatesDualNumberOfAVectorsAndScalar(double[] aValues,
                                                               double bValue,
                                                               DoubleTensor expectedGradientWrtA,
                                                               DoubleTensor expectedGradientWrtB,
                                                               BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        UniformVertex A = new UniformVertex(new int[]{1, aValues.length}, 0.0, 1.0);
        A.setAndCascade(DoubleTensor.create(aValues));
        UniformVertex B = new UniformVertex(0.0, 1.0);
        B.setAndCascade(DoubleTensor.scalar(bValue));

        DualNumber result = op.apply(A, B).getDualNumber();

        DoubleTensor wrtA = result.getPartialDerivatives().withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtA.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtA.getShape());

        DoubleTensor wrtB = result.getPartialDerivatives().withRespectTo(B);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtB.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtB.getShape());
    }

    public static double[] toDiagonalArray(double[] diagonal) {
        return DoubleTensor.create(diagonal).diag().asFlatDoubleArray();
    }
}

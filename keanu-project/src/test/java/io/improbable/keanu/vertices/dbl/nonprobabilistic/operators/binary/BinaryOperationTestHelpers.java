package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.function.BiFunction;

import com.google.common.collect.ImmutableSet;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

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
        DoubleVertex output = op.apply(A, B);

        PartialDerivatives wrtForward = output.getDualNumber();
        assertEquals(expectedGradientWrtA, wrtForward.withRespectTo(A).scalar(), 1e-5);
        assertEquals(expectedGradientWrtB, wrtForward.withRespectTo(B).scalar(), 1e-5);

        PartialDerivatives wrtReverse = Differentiator.reverseModeAutoDiff(output, ImmutableSet.of(A, B));
        assertEquals(expectedGradientWrtA, wrtReverse.withRespectTo(A).scalar(), 1e-5);
        assertEquals(expectedGradientWrtB, wrtReverse.withRespectTo(B).scalar(), 1e-5);
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

    public static void calculatesDualNumberOfTwoMatricesElementWiseOperator(DoubleTensor aValues,
                                                                            DoubleTensor bValues,
                                                                            DoubleTensor expectedGradientWrtA,
                                                                            DoubleTensor expectedGradientWrtB,
                                                                            BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {

        UniformVertex A = new UniformVertex(aValues.getShape(), 0.0, 1.0);
        A.setAndCascade(aValues);
        UniformVertex B = new UniformVertex(bValues.getShape(), 0.0, 1.0);
        B.setAndCascade(bValues);

        DoubleVertex output = op.apply(A, B);
        PartialDerivatives wrtForward = output.getDualNumber();

        DoubleTensor wrtAForward = wrtForward.withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtAForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtAForward.getShape());

        DoubleTensor wrtBForward = wrtForward.withRespectTo(B);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtBForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtBForward.getShape());

        PartialDerivatives wrtReverse = Differentiator.reverseModeAutoDiff(output, ImmutableSet.of(A, B));

        DoubleTensor wrtAReverse = wrtReverse.withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtAReverse.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtAReverse.getShape());

        DoubleTensor wrtBReverse = wrtReverse.withRespectTo(B);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtBReverse.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtBReverse.getShape());
    }

    public static void calculatesDualNumberOfAVectorAndScalar(DoubleTensor aValues,
                                                              double bValue,
                                                              DoubleTensor expectedGradientWrtA,
                                                              DoubleTensor expectedGradientWrtB,
                                                              BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        UniformVertex A = new UniformVertex(aValues.getShape(), 0.0, 1.0);
        A.setAndCascade(aValues);
        UniformVertex B = new UniformVertex(0.0, 1.0);
        B.setAndCascade(DoubleTensor.scalar(bValue));

        DoubleVertex output = op.apply(A, B);
        PartialDerivatives wrtForward = output.getDualNumber();

        DoubleTensor wrtAForward = wrtForward.withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtAForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtAForward.getShape());

        DoubleTensor wrtBForward = wrtForward.withRespectTo(B);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtBForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtBForward.getShape());

        PartialDerivatives wrtReverse = Differentiator.reverseModeAutoDiff(output, ImmutableSet.of(A, B));
        DoubleTensor wrtAReverse = wrtReverse.withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtAReverse.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtAReverse.getShape());

        DoubleTensor wrtBReverse = wrtReverse.withRespectTo(B);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtBReverse.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtBReverse.getShape());
    }

    public static void calculatesDualNumberOfAScalarAndVector(double aValue,
                                                              DoubleTensor bValues,
                                                              DoubleTensor expectedGradientWrtA,
                                                              DoubleTensor expectedGradientWrtB,
                                                              BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        UniformVertex A = new UniformVertex(0.0, 1.0);
        A.setAndCascade(DoubleTensor.scalar(aValue));
        UniformVertex B = new UniformVertex(bValues.getShape(), 0.0, 1.0);
        B.setAndCascade(bValues);

        DoubleVertex output = op.apply(A, B);
        PartialDerivatives wrtForward = output.getDualNumber();

        DoubleTensor wrtAForward = wrtForward.withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtAForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtAForward.getShape());

        DoubleTensor wrtBForward = wrtForward.withRespectTo(B);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtBForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtBForward.getShape());

        PartialDerivatives wrtReverse = Differentiator.reverseModeAutoDiff(output, ImmutableSet.of(A, B));
        DoubleTensor wrtAReverse = wrtReverse.withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtAReverse.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtAReverse.getShape());

        DoubleTensor wrtBReverse = wrtReverse.withRespectTo(B);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtBReverse.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtBReverse.getShape());
    }

    public static double[] toDiagonalArray(double[] diagonal) {
        return DoubleTensor.create(diagonal).diag().asFlatDoubleArray();
    }
}

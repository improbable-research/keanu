package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsOf;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import java.util.function.BiFunction;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class BinaryOperationTestHelpers {

    public static void operatesOnTwoScalarVertexValues(double aValue,
                                                       double bValue,
                                                       double expected,
                                                       BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {

        UniformVertex A = new UniformVertex(0.0, 1.0);
        A.setAndCascade(DoubleTensor.scalar(aValue));
        UniformVertex B = new UniformVertex(0.0, 1.0);
        B.setAndCascade(DoubleTensor.scalar(bValue));

        assertEquals(expected, op.apply(A, B).getValue().scalar(), 1e-5);
    }

    public static <T extends DoubleVertex & Differentiable>
    void calculatesDerivativeOfTwoScalars(double aValue,
                                          double bValue,
                                          double expectedGradientWrtA,
                                          double expectedGradientWrtB,
                                          BiFunction<DoubleVertex, DoubleVertex, T> op) {

        UniformVertex A = new UniformVertex(0.0, 1.0);
        A.setAndCascade(DoubleTensor.scalar(aValue));
        UniformVertex B = new UniformVertex(0.0, 1.0);
        B.setAndCascade(DoubleTensor.scalar(bValue));
        T output = op.apply(A, B);

        assertEquals(expectedGradientWrtA, Differentiator.forwardModeAutoDiff(A, output).of(output).scalar(), 1e-5);
        assertEquals(expectedGradientWrtB, Differentiator.forwardModeAutoDiff(B, output).of(output).scalar(), 1e-5);

        PartialsOf wrtReverse = Differentiator.reverseModeAutoDiff(output, ImmutableSet.of(A, B));
        assertEquals(expectedGradientWrtA, wrtReverse.withRespectTo(A).scalar(), 1e-5);
        assertEquals(expectedGradientWrtB, wrtReverse.withRespectTo(B).scalar(), 1e-5);
    }

    public static void operatesOnTwo2x2MatrixVertexValues(double[] aValues,
                                                          double[] bValues,
                                                          double[] expected,
                                                          BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {

        UniformVertex A = new UniformVertex(new long[]{2, 2}, 0.0, 1.0);
        A.setAndCascade(DoubleTensor.create(aValues, new long[]{2, 2}));
        UniformVertex B = new UniformVertex(new long[]{2, 2}, 0.0, 1.0);
        B.setAndCascade(DoubleTensor.create(bValues, new long[]{2, 2}));

        DoubleTensor result = op.apply(A, B).getValue();

        DoubleTensor expectedTensor = DoubleTensor.create(expected, new long[]{2, 2});

        assertEquals(expectedTensor.getValue(0, 0), result.getValue(0, 0), 1e-5);
        assertEquals(expectedTensor.getValue(0, 1), result.getValue(0, 1), 1e-5);
        assertEquals(expectedTensor.getValue(1, 0), result.getValue(1, 0), 1e-5);
        assertEquals(expectedTensor.getValue(1, 1), result.getValue(1, 1), 1e-5);
    }

    public static <T extends DoubleVertex & Differentiable>
    void calculatesDerivativeOfTwoMatricesElementWiseOperator(DoubleTensor aValues,
                                                              DoubleTensor bValues,
                                                              DoubleTensor expectedGradientWrtA,
                                                              DoubleTensor expectedGradientWrtB,
                                                              BiFunction<DoubleVertex, DoubleVertex, T> op) {

        UniformVertex A = new UniformVertex(aValues.getShape(), 0.0, 1.0);
        A.setAndCascade(aValues);
        UniformVertex B = new UniformVertex(bValues.getShape(), 0.0, 1.0);
        B.setAndCascade(bValues);

        T output = op.apply(A, B);

        DoubleTensor wrtAForward = Differentiator.forwardModeAutoDiff(A, output).of(output);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtAForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtAForward.getShape());

        DoubleTensor wrtBForward = Differentiator.forwardModeAutoDiff(B, output).of(output);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtBForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtBForward.getShape());

        PartialsOf wrtReverse = Differentiator.reverseModeAutoDiff(output, ImmutableSet.of(A, B));

        DoubleTensor wrtAReverse = wrtReverse.withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtAReverse.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtAReverse.getShape());

        DoubleTensor wrtBReverse = wrtReverse.withRespectTo(B);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtBReverse.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtBReverse.getShape());
    }

    public static <T extends DoubleVertex & Differentiable>
    void calculatesDerivativeOfAVectorAndScalar(DoubleTensor aValues,
                                                double bValue,
                                                DoubleTensor expectedGradientWrtA,
                                                DoubleTensor expectedGradientWrtB,
                                                BiFunction<DoubleVertex, DoubleVertex, T> op) {
        UniformVertex A = new UniformVertex(aValues.getShape(), 0.0, 1.0);
        A.setAndCascade(aValues);
        UniformVertex B = new UniformVertex(0.0, 1.0);
        B.setAndCascade(DoubleTensor.scalar(bValue));

        T output = op.apply(A, B);

        DoubleTensor wrtAForward = Differentiator.forwardModeAutoDiff(A, output).of(output);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtAForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtAForward.getShape());

        DoubleTensor wrtBForward = Differentiator.forwardModeAutoDiff(B, output).of(output);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtBForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtBForward.getShape());

        PartialsOf wrtReverse = Differentiator.reverseModeAutoDiff(output, ImmutableSet.of(A, B));
        DoubleTensor wrtAReverse = wrtReverse.withRespectTo(A);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtAReverse.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtAReverse.getShape());

        DoubleTensor wrtBReverse = wrtReverse.withRespectTo(B);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtBReverse.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtBReverse.getShape());
    }

    public static <T extends DoubleVertex & Differentiable>
    void calculatesDerivativeOfAScalarAndVector(double aValue,
                                                DoubleTensor bValues,
                                                DoubleTensor expectedGradientWrtA,
                                                DoubleTensor expectedGradientWrtB,
                                                BiFunction<DoubleVertex, DoubleVertex, T> op) {
        UniformVertex A = new UniformVertex(0.0, 1.0);
        A.setAndCascade(DoubleTensor.scalar(aValue));
        UniformVertex B = new UniformVertex(bValues.getShape(), 0.0, 1.0);
        B.setAndCascade(bValues);

        T output = op.apply(A, B);

        DoubleTensor wrtAForward = Differentiator.forwardModeAutoDiff(A, output).of(output);
        assertArrayEquals(expectedGradientWrtA.asFlatDoubleArray(), wrtAForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtA.getShape(), wrtAForward.getShape());

        DoubleTensor wrtBForward = Differentiator.forwardModeAutoDiff(B, output).of(output);
        assertArrayEquals(expectedGradientWrtB.asFlatDoubleArray(), wrtBForward.asFlatDoubleArray(), 1e-10);
        assertArrayEquals(expectedGradientWrtB.getShape(), wrtBForward.getShape());

        PartialsOf wrtReverse = Differentiator.reverseModeAutoDiff(output, ImmutableSet.of(A, B));
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

    public static <T extends DoubleVertex & Differentiable> void finiteDifferenceMatchesElementwise(BiFunction<UniformVertex, UniformVertex, T> op) {
        testWithFiniteDifference(op, new long[0], new long[0]);
        testWithFiniteDifference(op, new long[]{3}, new long[]{3});
        testWithFiniteDifference(op, new long[]{2, 3}, new long[]{2, 3});
        testWithFiniteDifference(op, new long[]{2, 2, 2}, new long[]{2, 2, 2});
    }

    public static <T extends DoubleVertex & Differentiable> void finiteDifferenceMatchesBroadcast(BiFunction<UniformVertex, UniformVertex, T> op) {
        testWithFiniteDifference(op, new long[]{2, 2, 2}, new long[]{1, 1, 1});
        testWithFiniteDifference(op, new long[]{1, 1, 1}, new long[]{2, 2, 2});
        testWithFiniteDifference(op, new long[]{2, 2, 2}, new long[]{});
        testWithFiniteDifference(op, new long[]{}, new long[]{2, 2, 2});
        testWithFiniteDifference(op, new long[]{2, 4}, new long[]{4});
        testWithFiniteDifference(op, new long[]{2, 4}, new long[]{1, 4});
        testWithFiniteDifference(op, new long[]{4}, new long[]{2, 4});
        testWithFiniteDifference(op, new long[]{1, 4}, new long[]{2, 4});
        testWithFiniteDifference(op, new long[]{2, 1, 4}, new long[]{1, 1, 4});
        testWithFiniteDifference(op, new long[]{2, 3, 4}, new long[]{1, 3, 4});
        testWithFiniteDifference(op, new long[]{2, 3, 4}, new long[]{3, 4});
        testWithFiniteDifference(op, new long[]{3, 4}, new long[]{2, 3, 4});
        testWithFiniteDifference(op, new long[]{2, 3, 4}, new long[]{4});
        testWithFiniteDifference(op, new long[]{4}, new long[]{2, 3, 4});
    }

    public static <T extends DoubleVertex & Differentiable> void testWithFiniteDifference(BiFunction<UniformVertex, UniformVertex, T> op,
                                                                                          long[] leftShape,
                                                                                          long[] rightShape) {
        UniformVertex A = new UniformVertex(leftShape, -10.0, 10.0);
        UniformVertex B = new UniformVertex(rightShape, -10.0, 10.0);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(A, B), op.apply(A, B), 1e-10, 1e-10);
    }
}

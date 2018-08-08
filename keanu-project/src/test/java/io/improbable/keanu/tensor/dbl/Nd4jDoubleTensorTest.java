package io.improbable.keanu.tensor.dbl;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class Nd4jDoubleTensorTest {

    Nd4jDoubleTensor matrixA;
    Nd4jDoubleTensor matrixB;
    Nd4jDoubleTensor matrixC;
    Nd4jDoubleTensor scalarA;

    @Before
    public void setup() {
        matrixA = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        matrixB = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        matrixC = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, new int[]{4, 4});
        scalarA = Nd4jDoubleTensor.scalar(2.0);
    }

    @Test
    public void canElementWiseMultiplyMatrix() {
        INDArray x = Nd4j.diag(Nd4j.arange(0, 16).reshape(4, 4));
        DoubleTensor result = matrixA.times(matrixB);
        assertArrayEquals(new double[]{1, 4, 9, 16}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canMultiplyMatrixByScalar() {
        DoubleTensor result = matrixA.times(scalarA);
        assertArrayEquals(new double[]{2, 4, 6, 8}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canMultiplyScalarByMatrix() {
        DoubleTensor result = scalarA.times(matrixA);
        assertArrayEquals(new double[]{2, 4, 6, 8}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canElementWiseMultiplyMatrixInPlace() {
        DoubleTensor result = matrixA.timesInPlace(matrixB);
        assertArrayEquals(new double[]{1, 4, 9, 16}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canMultiplyMatrixByScalarInPlace() {
        DoubleTensor result = matrixA.timesInPlace(scalarA);
        assertArrayEquals(new double[]{2, 4, 6, 8}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canElementWiseDivideMatrix() {
        DoubleTensor result = matrixA.div(matrixB);
        assertArrayEquals(new double[]{1.0, 1.0, 1.0, 1.0}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canDivideMatrixByScalar() {
        DoubleTensor result = matrixA.div(scalarA);
        assertArrayEquals(new double[]{1.0 / 2.0, 2.0 / 2.0, 3.0 / 2.0, 4.0 / 2.0}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canDivideScalarByMatrix() {
        DoubleTensor result = scalarA.div(matrixA);
        assertArrayEquals(new double[]{2.0 / 1.0, 2.0 / 2.0, 2.0 / 3.0, 2.0 / 4.0}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereGreaterThanAMatrix() {
        DoubleTensor mask = matrixA.getGreaterThanMask(Nd4jDoubleTensor.create(new double[]{2, 2, 2, 2}, new int[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2);

        assertArrayEquals(new double[]{1, 2, -2, -2}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereGreaterThanAScalar() {
        DoubleTensor mask = matrixA.getGreaterThanMask(Nd4jDoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2);

        assertArrayEquals(new double[]{1, 2, -2, -2}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanOrEqualAMatrix() {
        DoubleTensor mask = matrixA.getLessThanOrEqualToMask(Nd4jDoubleTensor.create(new double[]{2, 2, 2, 2}, new int[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2);

        assertArrayEquals(new double[]{-2, -2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanOrEqualAScalar() {
        DoubleTensor mask = matrixA.getLessThanOrEqualToMask(Nd4jDoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2);

        assertArrayEquals(new double[]{-2, -2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanAMatrix() {
        DoubleTensor mask = matrixA.getLessThanMask(Nd4jDoubleTensor.create(new double[]{2, 2, 2, 2}, new int[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2);

        assertArrayEquals(new double[]{-2, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanAScalar() {
        DoubleTensor mask = matrixA.getLessThanMask(Nd4jDoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2);

        assertArrayEquals(new double[]{-2, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canApplyUnaryFunctionToScalar() {
        DoubleTensor result = scalarA.apply(a -> a * 2);
        assertEquals(4, result.scalar(), 0.0);
    }

    @Test
    public void canApplyUnaryFunctionToRank3() {
        DoubleTensor rank3Tensor = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 2, 2});
        DoubleTensor result = rank3Tensor.apply(a -> a * 2);
        assertArrayEquals(new double[]{2, 4, 6, 8, 10, 12, 14, 16}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canApplySqrt() {
        DoubleTensor result = scalarA.sqrt();
        assertEquals(Math.sqrt(2.0), result.scalar(), 0.0);
    }

    @Test
    public void canBroadcastMultiplyRank4ContainingVectorAndMatrix() {
        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 2, 2, 1});
        DoubleTensor matrix = matrixA.reshape(2, 2, 1, 1);
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{1, 2, 6, 8, 15, 18, 28, 32}, new int[]{2, 2, 2, 1});

        assertTimesOperationEquals(rank4, matrix, expected);
        assertTimesInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyRank4ContainingMatrixAndMatrix() {
        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8,
            4, 3, 2, 1, 7, 5, 8, 6
        }, new int[]{2, 2, 2, 2});

        DoubleTensor matrix = matrixA.reshape(2, 2, 1, 1);
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 10, 12, 14, 16,
            12, 9, 6, 3, 28, 20, 32, 24
        }, new int[]{2, 2, 2, 2});


        assertTimesOperationEquals(rank4, matrix, expected);
        assertTimesInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyRank5ContainingMatrixAndMatrix() {
        DoubleTensor rank5 = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, 7, 5, 8, 6,
            6, 3, 2, 9, 3, 4, 7, 6, 6, 2, 5, 4, 0, 2, 1, 3
        }, new int[]{2, 2, 2, 2, 2});

        DoubleTensor matrix = matrixA.reshape(2, 2, 1, 1, 1);
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 8, 6, 4, 2, 14, 10, 16, 12,
            18, 9, 6, 27, 9, 12, 21, 18, 24, 8, 20, 16, 0, 8, 4, 12
        }, new int[]{2, 2, 2, 2, 2});

        assertTimesOperationEquals(rank5, matrix, expected);
        assertTimesInPlaceOperationEquals(rank5, matrix, expected);
    }

    @Test
    public void doesClampTensor() {
        DoubleTensor A = Nd4jDoubleTensor.create(new double[]{0.25, 3, -4, -5}, new int[]{1, 4});
        DoubleTensor clampedA = A.clamp(DoubleTensor.scalar(-4.5), DoubleTensor.scalar(2.0));
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{0.25, 2.0, -4.0, -4.5}, new int[]{1, 4});
        assertEquals(expected, clampedA);
    }

    @Test
    public void canBroadcastAdd() {
        DoubleTensor x = Nd4jDoubleTensor.create(new double[]{1, 2, 3}, new int[]{3, 1});
        DoubleTensor s = Nd4jDoubleTensor.create(new double[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new int[]{3, 5});

        DoubleTensor diff = s.plus(x);

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new int[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastSubtract() {
        DoubleTensor x = Nd4jDoubleTensor.create(new double[]{-1, -2, -3}, new int[]{3, 1});
        DoubleTensor s = Nd4jDoubleTensor.create(new double[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new int[]{3, 5});

        DoubleTensor diff = s.minus(x);

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new int[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastDivide() {
        DoubleTensor x = Nd4jDoubleTensor.create(new double[]{1, 2, 3}, new int[]{3, 1});
        DoubleTensor s = Nd4jDoubleTensor.create(new double[]{
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8
        }, new int[]{3, 5});

        DoubleTensor diff = s.div(x);

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            5 / 1.0, 2 / 1.0, 3 / 1.0, 7 / 1.0, 8 / 1.0,
            5 / 2.0, 2 / 2.0, 3 / 2.0, 7 / 2.0, 8 / 2.0,
            5 / 3.0, 2 / 3.0, 3 / 3.0, 7 / 3.0, 8 / 3.0
        }, new int[]{3, 5});

        assertEquals(expected, diff);
    }

    private void assertTimesOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        DoubleTensor actual = left.times(right);
        assertEquals(actual, expected);
    }

    private void assertTimesInPlaceOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        left.timesInPlace(right);
        assertEquals(left, expected);
    }

}

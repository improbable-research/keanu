package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;

public class Nd4jIntegerTensorTest {

    @Test
    public void doesMinusScalar() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor result = matrixA.minus(2);
        int[] expected = new int[]{-1, 0, 1, 2};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.minusInPlace(2);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesPlusScalar() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor result = matrixA.plus(2);
        int[] expected = new int[]{3, 4, 5, 6};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.plusInPlace(2);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesTimesScalar() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor result = matrixA.times(2);
        int[] expected = new int[]{2, 4, 6, 8};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.timesInPlace(2);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesDivideScalar() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor result = matrixA.div(2);
        int[] expected = new int[]{0, 1, 1, 2};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.divInPlace(2);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
        assertArrayEquals(new double[]{0.0, 1.0, 1.0, 2.0}, matrixA.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void doesElementwisePower() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixB = IntegerTensor.create(new int[]{2, 3, 2, 0}, new int[]{2, 2});
        IntegerTensor result = matrixA.pow(matrixB);
        int[] expected = new int[]{1, 8, 9, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.powInPlace(matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesScalarPower() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor result = matrixA.pow(2);
        int[] expected = new int[]{1, 4, 9, 16};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.powInPlace(2);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseMinus() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixB = IntegerTensor.create(new int[]{2, 3, 2, 0}, new int[]{2, 2});
        IntegerTensor result = matrixA.minus(matrixB);
        int[] expected = new int[]{-1, -1, 1, 4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.minusInPlace(matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwisePlus() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixB = IntegerTensor.create(new int[]{2, 3, 2, 0}, new int[]{2, 2});
        IntegerTensor result = matrixA.plus(matrixB);
        int[] expected = new int[]{3, 5, 5, 4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.plusInPlace(matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseTimes() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixB = IntegerTensor.create(new int[]{2, 3, 2, 0}, new int[]{2, 2});
        IntegerTensor result = matrixA.times(matrixB);
        int[] expected = new int[]{2, 6, 6, 0};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.timesInPlace(matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseDivide() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixC = IntegerTensor.create(new int[]{5, -1, 7, 2}, new int[]{2, 2});
        IntegerTensor result = matrixA.div(matrixC);
        int[] expected = new int[]{0, -2, 0, 2};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.divInPlace(matrixC);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
        assertArrayEquals(new double[]{0.0, -2.0, 0.0, 2.0}, matrixA.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void doesElementwiseUnaryMinus() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor result = matrixA.unaryMinus();
        int[] expected = new int[]{-1, -2, -3, -4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.unaryMinusInPlace();
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseAbsolute() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{-1, -2, 3, 4}, new int[]{2, 2});
        IntegerTensor result = matrixA.abs();
        int[] expected = new int[]{1, 2, 3, 4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.absInPlace();
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseGreaterThanMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{-1, -2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixTwos = IntegerTensor.create(2, new int[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getGreaterThanMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getGreaterThanMask(scalarTwo);

        int[] expected = new int[]{0, 0, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseGreaterThanOrEqualToMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{-1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixTwos = IntegerTensor.create(2, new int[]{2, 2});
        IntegerTensor scalarTWo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getGreaterThanOrEqualToMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getGreaterThanOrEqualToMask(scalarTWo);

        int[] expected = new int[]{0, 1, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{-1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixTwos = IntegerTensor.create(2, new int[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getLessThanMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getLessThanMask(scalarTwo);

        int[] expected = new int[]{1, 0, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanOrEqualToMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{-1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixTwos = IntegerTensor.create(2, new int[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getLessThanOrEqualToMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getLessThanOrEqualToMask(scalarTwo);

        int[] expected = new int[]{1, 1, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesSetWithMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{-1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor mask = IntegerTensor.create(new int[]{1, 1, 0, 0}, new int[]{2, 2});
        int[] expected = new int[]{100, 100, 3, 4};

        IntegerTensor result = matrix.setWithMask(mask, 100);
        assertArrayEquals(expected, result.asFlatIntegerArray());

        IntegerTensor resultInPlace = matrix.setWithMaskInPlace(mask, 100);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrix.asFlatIntegerArray());
    }

    @Test
    public void doesApplyUnaryFunction() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor result = matrixA.apply(v -> v + 1);
        int[] expected = new int[]{2, 3, 4, 5};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        IntegerTensor resultInPlace = matrixA.applyInPlace(v -> v + 1);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesCompareLessThanScalar() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        BooleanTensor result = matrix.lessThan(3);
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqualScalar() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual(3);
        Boolean[] expected = new Boolean[]{true, true, true, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThan() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor otherMatrix = IntegerTensor.create(new int[]{0, 2, 4, 7}, new int[]{2, 2});
        BooleanTensor result = matrix.lessThan(otherMatrix);
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqual() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor otherMatrix = IntegerTensor.create(new int[]{0, 2, 4, 7}, new int[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual(otherMatrix);
        Boolean[] expected = new Boolean[]{false, true, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThan() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor otherMatrix = IntegerTensor.create(new int[]{0, 2, 4, 7}, new int[]{2, 2});
        BooleanTensor result = matrix.greaterThan(otherMatrix);
        Boolean[] expected = new Boolean[]{true, false, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqual() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor otherMatrix = IntegerTensor.create(new int[]{0, 2, 4, 7}, new int[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(otherMatrix);
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanScalar() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        BooleanTensor result = matrix.greaterThan(3);
        Boolean[] expected = new Boolean[]{false, false, false, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalar() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(3);
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

}

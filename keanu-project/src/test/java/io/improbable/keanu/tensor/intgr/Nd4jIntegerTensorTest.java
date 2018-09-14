package io.improbable.keanu.tensor.intgr;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.util.Arrays;
import java.util.function.Function;

import org.junit.Test;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;

public class Nd4jIntegerTensorTest {

    @Test
    public void doesMinusScalar() {
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixB = Nd4jIntegerTensor.create(new int[]{2, 3, 2, 0}, new int[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixB = Nd4jIntegerTensor.create(new int[]{2, 3, 2, 0}, new int[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixB = Nd4jIntegerTensor.create(new int[]{2, 3, 2, 0}, new int[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixB = Nd4jIntegerTensor.create(new int[]{2, 3, 2, 0}, new int[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixC = Nd4jIntegerTensor.create(new int[]{5, -1, 7, 2}, new int[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{-1, -2, 3, 4}, new int[]{2, 2});
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
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{-1, -2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixTwos = Nd4jIntegerTensor.create(2, new int[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getGreaterThanMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getGreaterThanMask(scalarTwo);

        int[] expected = new int[]{0, 0, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseGreaterThanOrEqualToMask() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{-1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixTwos = Nd4jIntegerTensor.create(2, new int[]{2, 2});
        IntegerTensor scalarTWo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getGreaterThanOrEqualToMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getGreaterThanOrEqualToMask(scalarTWo);

        int[] expected = new int[]{0, 1, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanMask() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{-1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixTwos = Nd4jIntegerTensor.create(2, new int[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getLessThanMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getLessThanMask(scalarTwo);

        int[] expected = new int[]{1, 0, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanOrEqualToMask() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{-1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor matrixTwos = Nd4jIntegerTensor.create(2, new int[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getLessThanOrEqualToMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getLessThanOrEqualToMask(scalarTwo);

        int[] expected = new int[]{1, 1, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesSetWithMask() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{-1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor mask = Nd4jIntegerTensor.create(new int[]{1, 1, 0, 0}, new int[]{2, 2});
        int[] expected = new int[]{100, 100, 3, 4};

        IntegerTensor result = matrix.setWithMask(mask, 100);
        assertArrayEquals(expected, result.asFlatIntegerArray());

        IntegerTensor resultInPlace = matrix.setWithMaskInPlace(mask, 100);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrix.asFlatIntegerArray());
    }

    @Test
    public void doesApplyUnaryFunction() {
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
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
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        BooleanTensor result = matrix.lessThan(3);
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqualScalar() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual(3);
        Boolean[] expected = new Boolean[]{true, true, true, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThan() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor otherMatrix = Nd4jIntegerTensor.create(new int[]{0, 2, 4, 7}, new int[]{2, 2});
        BooleanTensor result = matrix.lessThan(otherMatrix);
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqual() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor otherMatrix = Nd4jIntegerTensor.create(new int[]{0, 2, 4, 7}, new int[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual(otherMatrix);
        Boolean[] expected = new Boolean[]{false, true, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThan() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor otherMatrix = Nd4jIntegerTensor.create(new int[]{0, 2, 4, 7}, new int[]{2, 2});
        BooleanTensor result = matrix.greaterThan(otherMatrix);
        Boolean[] expected = new Boolean[]{true, false, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqual() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor otherMatrix = Nd4jIntegerTensor.create(new int[]{0, 2, 4, 7}, new int[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(otherMatrix);
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanScalar() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        BooleanTensor result = matrix.greaterThan(3);
        Boolean[] expected = new Boolean[]{false, false, false, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalar() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(3);
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalarTensor() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(Nd4jIntegerTensor.scalar(3));
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void canBroadcastAdd() {
        IntegerTensor x = Nd4jIntegerTensor.create(new int[]{1, 2, 3}, new int[]{3, 1});
        IntegerTensor s = Nd4jIntegerTensor.create(new int[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new int[]{3, 5});

        IntegerTensor diff = s.plus(x);

        IntegerTensor expected = Nd4jIntegerTensor.create(new int[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new int[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastSubtract() {
        IntegerTensor x = Nd4jIntegerTensor.create(new int[]{-1, -2, -3}, new int[]{3, 1});
        IntegerTensor s = Nd4jIntegerTensor.create(new int[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new int[]{3, 5});

        IntegerTensor diff = s.minus(x);

        IntegerTensor expected = Nd4jIntegerTensor.create(new int[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new int[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastDivide() {
        IntegerTensor x = Nd4jIntegerTensor.create(new int[]{1, 2, 3}, new int[]{3, 1});
        IntegerTensor s = Nd4jIntegerTensor.create(new int[]{
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8
        }, new int[]{3, 5});

        IntegerTensor diff = s.div(x);

        IntegerTensor expected = Nd4jIntegerTensor.create(new int[]{
            5 / 1, 2 / 1, 3 / 1, 7 / 1, 8 / 1,
            5 / 2, 2 / 2, 3 / 2, 7 / 2, 8 / 2,
            5 / 3, 2 / 3, 3 / 3, 7 / 3, 8 / 3
        }, new int[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void doesPositiveDivisionCorrectly() {
        assertDropsFractionCorrectlyOnDivision(5, 3);
    }

    @Test
    public void doesNegativeDivisionCorrectly() {
        assertDropsFractionCorrectlyOnDivision(-5, 3);
    }

    @Test
    public void canStartAsMaxInteger() {
        assertDropsFractionCorrectlyOnDivision(Integer.MAX_VALUE, 3);
    }

    @Test
    public void canStartAsMinInteger() {
        assertDropsFractionCorrectlyOnDivision(Integer.MIN_VALUE, 3);
    }

    private void assertDropsFractionCorrectlyOnDivision(int numerator, int denominator) {
        int expected = numerator / denominator;
        IntegerTensor tensor = IntegerTensor.create(
            new int[]{numerator, numerator, numerator, numerator},
            new int[]{2, 2}
        );
        IntegerTensor result = tensor.div(denominator);
        assertArrayEquals(new int[]{expected, expected, expected, expected}, result.asFlatIntegerArray());
    }

    @Test
    public void canResultInMaxInteger() {
        IntegerTensor tensor = IntegerTensor.create(
            new int[]{0, 0, 0, 0},
            new int[]{2, 2}
        );
        IntegerTensor result = tensor.plus(Integer.MAX_VALUE);
        int expected = Integer.MAX_VALUE;
        assertArrayEquals(new int[]{expected, expected, expected, expected}, result.asFlatIntegerArray());
    }

    @Test
    public void canResultInMinInteger() {
        IntegerTensor tensor = IntegerTensor.create(
            new int[]{0, 0, 0, 0},
            new int[]{2, 2}
        );
        IntegerTensor result = tensor.plus(Integer.MIN_VALUE);
        int expected = Integer.MIN_VALUE;
        assertArrayEquals(new int[]{expected, expected, expected, expected}, result.asFlatIntegerArray());
    }

    @Test
    public void canRepresentAllValues() {
        IntegerTensor tensor = IntegerTensor.create(
            new int[]{0,0,0,0},
            new int[]{2, 2}
        );

        /*
         * Construct a value that has the most significant two bits and the least significant bit set to 1 with all
         * others set to 0.  This value will stretch any floating point backing representation by requiring at least a
         * <Num bits> - 1 length Mantissa.  Simply using INT MAX often doesn't work for this test as the closest
         * floating point value is usually > INT MAX and when converting back, the value will be clamped back to max
         */
        final int biggestBitRange = (0x3 << Integer.SIZE - 2) + 1;

        tensor.plusInPlace(biggestBitRange);
        assertArrayEquals(new int[]{biggestBitRange, biggestBitRange, biggestBitRange, biggestBitRange},
            tensor.asFlatIntegerArray());
    }

    @Test
    public void youCanCheckForZeros() {
        IntegerTensor containsZero = IntegerTensor.create(new int[]{
                0, -1, Integer.MAX_VALUE,
                Integer.MIN_VALUE, -0, 1},
            3, 2);

        BooleanTensor expectedMask = BooleanTensor.create(new boolean[]{
                false, true, true,
                true, false, true},
            3, 2);

        TensorValidator<Integer, IntegerTensor> validator = TensorValidator.thatChecksFor(0);
        assertThat(validator.check(containsZero), equalTo(expectedMask));
    }

    @Test
    public void youCanFixAValidationIssueByReplacingTheValue() {
        IntegerTensor containsZero = IntegerTensor.create(1, 0, -1);
        IntegerTensor expectedResult = IntegerTensor.create(1, 0, 0);

        TensorValidator validator = TensorValidator.thatChecksFor(-1).withPolicy(TensorValidationPolicy.changeValueTo(0));
        assertThat(validator.validate(containsZero), equalTo(expectedResult));
    }

    @Test
    public void youCanFixACustomValidationIssueByReplacingTheValue() {
        IntegerTensor containsZero = IntegerTensor.create(1, 0, -1);
        IntegerTensor expectedResult = IntegerTensor.create(1, 0, 0);

        Function<Integer, Boolean> checkFunction = x -> x >= 0;
        TensorValidator<Integer, IntegerTensor> validator = TensorValidator.thatExpects(checkFunction).withPolicy(TensorValidationPolicy.changeValueTo(0));
        assertThat(validator.validate(containsZero), equalTo(expectedResult));
    }
}

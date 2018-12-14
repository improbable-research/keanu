package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.TensorTestHelper;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;
import junit.framework.TestCase;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.Arrays;
import java.util.stream.IntStream;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;


public class Nd4jIntegerTensorTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Test
    public void youCanCreateARankZeroTensor() {
        IntegerTensor scalar = IntegerTensor.create(new int[]{2}, new long[]{});
        assertEquals(2, (int) scalar.scalar());
        TestCase.assertEquals(0, scalar.getRank());
    }

    @Test
    public void youCanCreateARankOneTensor() {
        IntegerTensor vector = IntegerTensor.create(new int[]{1, 2, 3, 4, 5}, new long[]{5});
        assertEquals(4, (int) vector.getValue(3));
        TestCase.assertEquals(1, vector.getRank());
    }

    @Test
    public void doesMinusScalar() {
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixB = Nd4jIntegerTensor.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixB = Nd4jIntegerTensor.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixB = Nd4jIntegerTensor.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixB = Nd4jIntegerTensor.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixC = Nd4jIntegerTensor.create(new int[]{5, -1, 7, 2}, new long[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{-1, -2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{-1, -2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixTwos = Nd4jIntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getGreaterThanMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getGreaterThanMask(scalarTwo);

        int[] expected = new int[]{0, 0, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseGreaterThanOrEqualToMask() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixTwos = Nd4jIntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor scalarTWo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getGreaterThanOrEqualToMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getGreaterThanOrEqualToMask(scalarTWo);

        int[] expected = new int[]{0, 1, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanMask() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixTwos = Nd4jIntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getLessThanMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getLessThanMask(scalarTwo);

        int[] expected = new int[]{1, 0, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanOrEqualToMask() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixTwos = Nd4jIntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.getLessThanOrEqualToMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.getLessThanOrEqualToMask(scalarTwo);

        int[] expected = new int[]{1, 1, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesSetWithMask() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor mask = Nd4jIntegerTensor.create(new int[]{1, 1, 0, 0}, new long[]{2, 2});
        int[] expected = new int[]{100, 100, 3, 4};

        IntegerTensor result = matrix.setWithMask(mask, 100);
        assertArrayEquals(expected, result.asFlatIntegerArray());

        IntegerTensor resultInPlace = matrix.setWithMaskInPlace(mask, 100);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrix.asFlatIntegerArray());
    }

    @Test
    public void cannotSetIfMaskLengthIsSmallerThanTensorLength() {
        IntegerTensor tensor = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor mask = Nd4jIntegerTensor.scalar(1);

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The lengths of the tensor and mask must match, but got tensor length: " + tensor.getLength() + ", mask length: " + mask.getLength());

        tensor.setWithMaskInPlace(mask, -2);
    }

    @Test
    public void cannotSetIfMaskLengthIsLargerThanTensorLength() {
        IntegerTensor tensor = Nd4jIntegerTensor.scalar(3);
        IntegerTensor mask = Nd4jIntegerTensor.create(new int[]{1, 1, 1, 1}, new long[]{2, 2});

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The lengths of the tensor and mask must match, but got tensor length: " + tensor.getLength() + ", mask length: " + mask.getLength());

        tensor.setWithMaskInPlace(mask, -2);
    }

    @Test
    public void doesApplyUnaryFunction() {
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThan(3);
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqualScalar() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual(3);
        Boolean[] expected = new Boolean[]{true, true, true, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThan() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor otherMatrix = Nd4jIntegerTensor.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThan(otherMatrix);
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqual() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor otherMatrix = Nd4jIntegerTensor.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual(otherMatrix);
        Boolean[] expected = new Boolean[]{false, true, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThan() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor otherMatrix = Nd4jIntegerTensor.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThan(otherMatrix);
        Boolean[] expected = new Boolean[]{true, false, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqual() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor otherMatrix = Nd4jIntegerTensor.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(otherMatrix);
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanScalar() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThan(3);
        Boolean[] expected = new Boolean[]{false, false, false, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalar() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(3);
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalarTensor() {
        IntegerTensor matrix = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(Nd4jIntegerTensor.scalar(3));
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }


    @Test
    public void canElementwiseEqualsAScalarValue() {
        int value = 42;
        int otherValue = 43;
        IntegerTensor allTheSame = IntegerTensor.create(value, new long[]{2, 3});
        IntegerTensor notAllTheSame = allTheSame.duplicate().setValue(otherValue, 1, 1);

        assertThat(allTheSame.elementwiseEquals(value).allTrue(), equalTo(true));
        assertThat(notAllTheSame.elementwiseEquals(value), hasValue(true, true, true, true, false, true));
    }

    @Test
    public void canBroadcastAdd() {
        IntegerTensor x = Nd4jIntegerTensor.create(new int[]{1, 2, 3}, new long[]{3, 1});
        IntegerTensor s = Nd4jIntegerTensor.create(new int[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new long[]{3, 5});

        IntegerTensor diff = s.plus(x);

        IntegerTensor expected = Nd4jIntegerTensor.create(new int[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new long[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastSubtract() {
        IntegerTensor x = Nd4jIntegerTensor.create(new int[]{-1, -2, -3}, new long[]{3, 1});
        IntegerTensor s = Nd4jIntegerTensor.create(new int[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new long[]{3, 5});

        IntegerTensor diff = s.minus(x);

        IntegerTensor expected = Nd4jIntegerTensor.create(new int[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new long[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastDivide() {
        IntegerTensor x = Nd4jIntegerTensor.create(new int[]{1, 2, 3}, new long[]{3, 1});
        IntegerTensor s = Nd4jIntegerTensor.create(new int[]{
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8
        }, new long[]{3, 5});

        IntegerTensor diff = s.div(x);

        IntegerTensor expected = Nd4jIntegerTensor.create(new int[]{
            5 / 1, 2 / 1, 3 / 1, 7 / 1, 8 / 1,
            5 / 2, 2 / 2, 3 / 2, 7 / 2, 8 / 2,
            5 / 3, 2 / 3, 3 / 3, 7 / 3, 8 / 3
        }, new long[]{3, 5});

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
            new long[]{2, 2}
        );
        IntegerTensor result = tensor.div(denominator);
        assertArrayEquals(new int[]{expected, expected, expected, expected}, result.asFlatIntegerArray());
    }

    @Test
    public void canResultInMaxInteger() {
        IntegerTensor tensor = IntegerTensor.create(
            new int[]{0, 0, 0, 0},
            new long[]{2, 2}
        );
        IntegerTensor result = tensor.plus(Integer.MAX_VALUE);
        int expected = Integer.MAX_VALUE;
        assertArrayEquals(new int[]{expected, expected, expected, expected}, result.asFlatIntegerArray());
    }

    @Test
    public void canResultInMinInteger() {
        IntegerTensor tensor = IntegerTensor.create(
            new int[]{0, 0, 0, 0},
            new long[]{2, 2}
        );
        IntegerTensor result = tensor.plus(Integer.MIN_VALUE);
        int expected = Integer.MIN_VALUE;
        assertArrayEquals(new int[]{expected, expected, expected, expected}, result.asFlatIntegerArray());
    }

    @Test
    public void canRepresentAllValues() {
        IntegerTensor tensor = IntegerTensor.create(
            new int[]{0, 0, 0, 0},
            new long[]{2, 2}
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
    public void canFindScalarMinAndMax() {
        IntegerTensor a = IntegerTensor.create(5, 4, 3, 2).reshape(2, 2);
        int min = a.min();
        int max = a.max();
        assertEquals(2, min);
        assertEquals(5, max);
    }

    @Test
    public void canFindMinAndMaxFromScalarToTensor() {
        IntegerTensor a = IntegerTensor.create(5, 4, 3, 2).reshape(1, 4);
        IntegerTensor b = IntegerTensor.create(3);

        IntegerTensor min = IntegerTensor.min(a, b);
        IntegerTensor max = IntegerTensor.max(a, b);

        assertArrayEquals(new int[]{3, 3, 3, 2}, min.asFlatIntegerArray());
        assertArrayEquals(new int[]{5, 4, 3, 3}, max.asFlatIntegerArray());
    }

    @Test
    public void canFindElementWiseMinAndMax() {
        IntegerTensor a = IntegerTensor.create(1, 2, 3, 4).reshape(1, 4);
        IntegerTensor b = IntegerTensor.create(2, 3, 1, 4).reshape(1, 4);

        IntegerTensor min = IntegerTensor.min(a, b);
        IntegerTensor max = IntegerTensor.max(a, b);

        assertArrayEquals(new int[]{1, 2, 1, 4}, min.asFlatIntegerArray());
        assertArrayEquals(new int[]{2, 3, 3, 4}, max.asFlatIntegerArray());
    }

    @Test
    public void canFindArgMaxOfRowVector() {
        IntegerTensor tensorRow = IntegerTensor.create(1, 3, 4, 5, 2).reshape(1, 5);

        assertEquals(3, tensorRow.argMax());
        assertThat(tensorRow.argMax(0), valuesAndShapesMatch(IntegerTensor.zeros(5)));
        assertThat(tensorRow.argMax(1), valuesAndShapesMatch(IntegerTensor.scalar(3)));
    }

    @Test
    public void canFindArgMaxOfColumnVector() {
        IntegerTensor tensorCol = IntegerTensor.create(1, 3, 4, 5, 2).reshape(5, 1);

        assertEquals(3, tensorCol.argMax());
        assertThat(tensorCol.argMax(0), valuesAndShapesMatch(IntegerTensor.scalar(3)));
        assertThat(tensorCol.argMax(1), valuesAndShapesMatch(IntegerTensor.zeros(5)));
    }

    @Test
    public void canFindArgMaxOfMatrix() {
        IntegerTensor tensor = IntegerTensor.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);

        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.create(1, 0, 0, 0)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.create(2, 0)));
        assertEquals(2, tensor.argMax());
    }

    @Test
    public void canFindArgMaxOfHighRank() {
        IntegerTensor tensor = IntegerTensor.create(IntStream.range(0, 512).toArray()).reshape(2, 8, 4, 2, 4);

        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.ones(8, 4, 2, 4)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.create(7, new long[]{2, 4, 2, 4})));
        assertThat(tensor.argMax(2), valuesAndShapesMatch(IntegerTensor.create(3, new long[]{2, 8, 2, 4})));
        assertThat(tensor.argMax(3), valuesAndShapesMatch(IntegerTensor.ones(2, 8, 4, 4)));
        assertEquals(511, tensor.argMax());
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHigh() {
        IntegerTensor tensor = IntegerTensor.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);
        tensor.argMax(2);
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

        TensorValidator<Integer, IntegerTensor> validator = TensorValidator.thatExpectsNotToFind(0);
        assertThat(validator.check(containsZero), equalTo(expectedMask));
    }

    @Test
    public void youCanFixAValidationIssueByReplacingTheValue() {
        IntegerTensor containsMinusOne = IntegerTensor.create(1, 0, -1);
        IntegerTensor expectedResult = IntegerTensor.create(1, 0, 0);

        TensorValidator<Integer, IntegerTensor> validator = TensorValidator.thatReplaces(-1, 0);
        containsMinusOne = validator.validate(containsMinusOne);
        assertThat(containsMinusOne, equalTo(expectedResult));
    }

    @Test
    public void youCanFixACustomValidationIssueByReplacingTheValue() {
        IntegerTensor containsMinusOne = IntegerTensor.create(1, 0, -1);
        IntegerTensor expectedResult = IntegerTensor.create(1, 0, 0);

        TensorValidator<Integer, IntegerTensor> validator = TensorValidator.thatFixesElementwise(x -> x >= 0, TensorValidationPolicy.changeValueTo(0));
        containsMinusOne = validator.validate(containsMinusOne);
        assertThat(containsMinusOne, equalTo(expectedResult));
    }

    @Test
    public void comparesIntegerTensorWithScalar() {
        IntegerTensor value = IntegerTensor.create(1, 2, 3);
        IntegerTensor differentValue = IntegerTensor.create(1);
        BooleanTensor result = value.elementwiseEquals(differentValue);
        assertThat(result, hasValue(true, false, false));
    }

    @Test
    public void canSliceRank3To2() {
        IntegerTensor x = IntegerTensor.create(1, 2, 3, 4, 1, 2, 3, 4).reshape(2, 2, 2);
        TensorTestHelper.doesDownRankOnSliceRank3To2(x);
    }

    @Test
    public void canSliceRank2To1() {
        IntegerTensor x = IntegerTensor.create(1, 2, 3, 4).reshape(2, 2);
        TensorTestHelper.doesDownRankOnSliceRank2To1(x);
    }

    @Test
    public void canSliceRank1ToScalar() {
        IntegerTensor x = IntegerTensor.create(1, 2, 3, 4).reshape(4);
        TensorTestHelper.doesDownRankOnSliceRank1ToScalar(x);
    }

    @Test
    public void canConcatScalars() {
        IntegerTensor x = IntegerTensor.scalar(2);
        IntegerTensor y = IntegerTensor.scalar(3);

        IntegerTensor concat = IntegerTensor.concat(x, y);
        assertEquals(IntegerTensor.create(2, 3), concat);
    }

    @Test
    public void canConcatVectors() {
        IntegerTensor x = IntegerTensor.create(2, 3);
        IntegerTensor y = IntegerTensor.create(4, 5);

        IntegerTensor concat = IntegerTensor.concat(x, y);
        assertEquals(IntegerTensor.create(2, 3, 4, 5), concat);
    }

    @Test
    public void canConcatMatrices() {
        IntegerTensor x = IntegerTensor.create(2, 3).reshape(1, 2);
        IntegerTensor y = IntegerTensor.create(4, 5).reshape(1, 2);

        IntegerTensor concat = IntegerTensor.concat(0, x, y);
        assertEquals(IntegerTensor.create(2, 3, 4, 5).reshape(2, 2), concat);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsWhenNeedsDimensionSpecifiedForConcat() {
        IntegerTensor x = IntegerTensor.create(2, 3).reshape(1, 2);
        IntegerTensor y = IntegerTensor.create(4, 5, 6).reshape(1, 3);

        IntegerTensor concat = IntegerTensor.concat(0, x, y);
        assertEquals(IntegerTensor.create(2, 3, 4, 5, 6), concat);
    }


}

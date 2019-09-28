package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.TensorFactories;
import io.improbable.keanu.tensor.TensorMatchers;
import io.improbable.keanu.tensor.TensorTestHelper;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;
import junit.framework.TestCase;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.stream.IntStream;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

@RunWith(Parameterized.class)
public class IntegerTensorTest {

    @Parameterized.Parameters(name = "{index}: Test with {1}")
    public static Iterable<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {new Nd4jIntegerTensorFactory(), "ND4J IntegerTensor"},
            {new JVMIntegerTensorFactory(), "JVM IntegerTensor"},
        });
    }

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    public IntegerTensorTest(IntegerTensorFactory factory, String name) {
        TensorFactories.integerTensorFactory = factory;
    }

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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixB = IntegerTensor.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixB = IntegerTensor.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixB = IntegerTensor.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixB = IntegerTensor.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixC = IntegerTensor.create(new int[]{5, -1, 7, 2}, new long[]{2, 2});
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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrixA = IntegerTensor.create(new int[]{-1, -2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrix = IntegerTensor.create(new int[]{-1, -2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixTwos = IntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.greaterThanMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.greaterThanMask(scalarTwo);

        int[] expected = new int[]{0, 0, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseGreaterThanOrEqualToMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixTwos = IntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor scalarTWo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.greaterThanOrEqualToMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.greaterThanOrEqualToMask(scalarTWo);

        int[] expected = new int[]{0, 1, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixTwos = IntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.lessThanMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.lessThanMask(scalarTwo);

        int[] expected = new int[]{1, 0, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanOrEqualToMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor matrixTwos = IntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);

        IntegerTensor maskFromMatrix = matrix.lessThanOrEqualToMask(matrixTwos);
        IntegerTensor maskFromScalar = matrix.lessThanOrEqualToMask(scalarTwo);

        int[] expected = new int[]{1, 1, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesSetWithMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor mask = IntegerTensor.create(new int[]{1, 1, 0, 0}, new long[]{2, 2});
        IntegerTensor expected = IntegerTensor.create(new int[]{100, 100, 3, 4}, new long[]{2, 2});

        IntegerTensor result = matrix.setWithMask(mask, 100);
        assertThat(result, valuesAndShapesMatch(expected));

        IntegerTensor resultInPlace = matrix.setWithMaskInPlace(mask, 100);
        assertThat(resultInPlace, valuesAndShapesMatch(expected));
        assertThat(matrix, valuesAndShapesMatch(expected));
    }

    @Test
    public void cannotCreateTensorWithLongsThatAreTooLong() {
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Out of range: " + Long.MAX_VALUE);

        IntegerTensor.create(new long[]{Long.MAX_VALUE});
    }

    @Test
    public void doesApplyUnaryFunction() {
        IntegerTensor matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
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
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThan(3);
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqualScalar() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual(3);
        Boolean[] expected = new Boolean[]{true, true, true, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThan() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor otherMatrix = IntegerTensor.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThan(otherMatrix);
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqual() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor otherMatrix = IntegerTensor.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual(otherMatrix);
        Boolean[] expected = new Boolean[]{false, true, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThan() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor otherMatrix = IntegerTensor.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThan(otherMatrix);
        Boolean[] expected = new Boolean[]{true, false, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqual() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor otherMatrix = IntegerTensor.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(otherMatrix);
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanScalar() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThan(3);
        Boolean[] expected = new Boolean[]{false, false, false, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalar() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(3);
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalarTensor() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(IntegerTensor.scalar(3));
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor otherMatrix = IntegerTensor.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        IntegerTensor result = matrix.greaterThanMask(otherMatrix);
        int[] expected = new int[]{1, 0, 0, 0};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor otherMatrix = IntegerTensor.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        IntegerTensor result = matrix.greaterThanOrEqualToMask(otherMatrix);
        int[] expected = new int[]{1, 1, 0, 0};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanScalarMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor result = matrix.greaterThanMask(IntegerTensor.scalar(3));
        int[] expected = new int[]{0, 0, 0, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalarMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor result = matrix.greaterThanOrEqualToMask(IntegerTensor.scalar(3));
        int[] expected = new int[]{0, 0, 1, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualTensorMask() {
        IntegerTensor matrix = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor result = matrix.greaterThanOrEqualToMask(IntegerTensor.create(0, 4));
        int[] expected = new int[]{1, 0, 1, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void canElementwiseEqualsAScalarValue() {
        int value = 42;
        int otherValue = 43;
        IntegerTensor allTheSame = IntegerTensor.create(value, new long[]{2, 3});
        IntegerTensor notAllTheSame = allTheSame.duplicate();
        notAllTheSame.setValue(otherValue, 1, 1);

        assertThat(allTheSame.elementwiseEquals(value).allTrue().scalar(), equalTo(true));
        assertThat(notAllTheSame.elementwiseEquals(value), hasValue(true, true, true, true, false, true));
    }

    @Test
    public void canBroadcastAdd() {
        IntegerTensor x = IntegerTensor.create(new int[]{1, 2, 3}, new long[]{3, 1});
        IntegerTensor s = IntegerTensor.create(new int[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new long[]{3, 5});

        IntegerTensor diff = s.plus(x);

        IntegerTensor expected = IntegerTensor.create(new int[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new long[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastSubtract() {
        IntegerTensor x = IntegerTensor.create(new int[]{-1, -2, -3}, new long[]{3, 1});
        IntegerTensor s = IntegerTensor.create(new int[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new long[]{3, 5});

        IntegerTensor diff = s.minus(x);

        IntegerTensor expected = IntegerTensor.create(new int[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new long[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastDivide() {
        IntegerTensor x = IntegerTensor.create(new int[]{1, 2, 3}, new long[]{3, 1});
        IntegerTensor s = IntegerTensor.create(new int[]{
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8
        }, new long[]{3, 5});

        IntegerTensor diff = s.div(x);

        IntegerTensor expected = IntegerTensor.create(new int[]{
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
        int min = a.min().scalar();
        int max = a.max().scalar();
        assertEquals(2, min);
        assertEquals(5, max);
    }

    @Test
    public void canFindMinAndMaxFromScalarToTensor() {
        IntegerTensor a = IntegerTensor.create(5, 4, 3, 2).reshape(1, 4);
        IntegerTensor b = IntegerTensor.scalar(3);

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

        assertThat(tensorRow.argMax().scalar(), equalTo(3));
        assertThat(tensorRow.argMax(0), valuesAndShapesMatch(IntegerTensor.zeros(5)));
        assertThat(tensorRow.argMax(1), valuesAndShapesMatch(IntegerTensor.create(new int[]{3}, 1)));
    }

    @Test
    public void canFindArgMaxOfColumnVector() {
        IntegerTensor tensorCol = IntegerTensor.create(1, 3, 4, 5, 2).reshape(5, 1);

        assertThat(tensorCol.argMax().scalar(), equalTo(3));
        assertThat(tensorCol.argMax(0), valuesAndShapesMatch(IntegerTensor.create(new int[]{3}, 1)));
        assertThat(tensorCol.argMax(1), valuesAndShapesMatch(IntegerTensor.zeros(5)));
    }

    @Test
    public void canFindArgMaxOfMatrix() {
        IntegerTensor tensor = IntegerTensor.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);

        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.create(1, 0, 0, 0)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.create(2, 0)));
        assertThat(tensor.argMax().scalar(), equalTo(2));
    }

    @Test
    public void canFindArgMinOfMatrix() {
        IntegerTensor tensor = IntegerTensor.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);

        assertThat(tensor.argMin(0), valuesAndShapesMatch(IntegerTensor.create(0, 1, 1, 1)));
        assertThat(tensor.argMin(1), valuesAndShapesMatch(IntegerTensor.create(0, 1)));
        assertThat(tensor.argMin().scalar(), equalTo(0));
    }

    @Test
    public void canFindArgMaxOfHighRank() {
        IntegerTensor tensor = IntegerTensor.create(IntStream.range(0, 512).toArray()).reshape(2, 8, 4, 2, 4);

        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.ones(8, 4, 2, 4)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.create(7, new long[]{2, 4, 2, 4})));
        assertThat(tensor.argMax(2), valuesAndShapesMatch(IntegerTensor.create(3, new long[]{2, 8, 2, 4})));
        assertThat(tensor.argMax(3), valuesAndShapesMatch(IntegerTensor.ones(2, 8, 4, 4)));
        assertThat(tensor.argMax().scalar(), equalTo(511));
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHigh() {
        IntegerTensor tensor = IntegerTensor.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);
        tensor.argMax(2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHighScalar() {
        IntegerTensor tensor = IntegerTensor.scalar(1);
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

        TensorValidator<Integer, IntegerTensor> validator = TensorValidator.thatFixesElementwise(x -> x >= 0, (TensorValidationPolicy<Integer, IntegerTensor>) TensorValidationPolicy.changeValueTo(0));
        containsMinusOne = validator.validate(containsMinusOne);
        assertThat(containsMinusOne, equalTo(expectedResult));
    }

    @Test
    public void comparesIntegerTensorWithScalar() {
        IntegerTensor value = IntegerTensor.create(1, 2, 3);
        IntegerTensor differentValue = IntegerTensor.scalar(1);
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

    @Test
    public void canStackScalars() {
        IntegerTensor x = IntegerTensor.scalar(2);
        IntegerTensor y = IntegerTensor.scalar(3);

        assertThat(IntegerTensor.create(2, 3).reshape(2), TensorMatchers.valuesAndShapesMatch(IntegerTensor.stack(0, x, y)));
    }

    @Test
    public void canStackVectors() {
        IntegerTensor x = IntegerTensor.create(2, 3);
        IntegerTensor y = IntegerTensor.create(4, 5);

        assertEquals(IntegerTensor.create(2, 3, 4, 5).reshape(2, 2), IntegerTensor.stack(0, x, y));
        assertEquals(IntegerTensor.create(2, 4, 3, 5).reshape(2, 2), IntegerTensor.stack(1, x, y));
    }

    @Test
    public void canStackMatrices() {
        IntegerTensor x = IntegerTensor.create(2, 3).reshape(1, 2);
        IntegerTensor y = IntegerTensor.create(4, 5).reshape(1, 2);

        assertEquals(IntegerTensor.create(2, 3, 4, 5).reshape(2, 1, 2), IntegerTensor.stack(0, x, y));
        assertEquals(IntegerTensor.create(2, 3, 4, 5).reshape(1, 2, 2), IntegerTensor.stack(1, x, y));
       /*
        Result in numpy when dimension is equal to array length:
        >>> a
        array([[2, 3]])
        >>> b
        array([[4, 5]])
        >>> np.stack([a, b], axis=2)
        array([[[2, 4],
                [3, 5]]])
        */
        assertEquals(IntegerTensor.create(2, 4, 3, 5).reshape(1, 2, 2), IntegerTensor.stack(2, x, y));
    }

    @Test
    public void canStackIfDimensionIsNegative() {
        IntegerTensor x = IntegerTensor.create(2, 3).reshape(1, 2);
        IntegerTensor y = IntegerTensor.create(4, 5).reshape(1, 2);

        assertThat(IntegerTensor.create(2, 3, 4, 5).reshape(2, 1, 2), valuesAndShapesMatch(IntegerTensor.stack(-3, x, y)));
        assertThat(IntegerTensor.create(2, 3, 4, 5).reshape(1, 2, 2), valuesAndShapesMatch(IntegerTensor.stack(-2, x, y)));
        assertThat(IntegerTensor.create(2, 4, 3, 5).reshape(1, 2, 2), valuesAndShapesMatch(IntegerTensor.stack(-1, x, y)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotStackIfPositiveDimensionIsOutOfBounds() {
        IntegerTensor x = IntegerTensor.create(2, 3).reshape(1, 2);
        IntegerTensor y = IntegerTensor.create(4, 5).reshape(1, 2);
        IntegerTensor.stack(3, x, y);
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotStackIfNegativeDimensionIsOutOfBounds() {
        IntegerTensor x = IntegerTensor.create(2, 3).reshape(1, 2);
        IntegerTensor y = IntegerTensor.create(4, 5).reshape(1, 2);
        IntegerTensor.stack(-4, x, y);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsWhenNeedsDimensionSpecifiedForConcat() {
        IntegerTensor x = IntegerTensor.create(2, 3).reshape(1, 2);
        IntegerTensor y = IntegerTensor.create(4, 5, 6).reshape(1, 3);

        IntegerTensor concat = IntegerTensor.concat(0, x, y);
        assertEquals(IntegerTensor.create(2, 3, 4, 5, 6), concat);
    }

    @Test
    public void canBroadcastToShape() {
        IntegerTensor a = IntegerTensor.create(
            1, 2, 3
        ).reshape(3);

        IntegerTensor expectedByRow = IntegerTensor.create(
            1, 2, 3,
            1, 2, 3,
            1, 2, 3
        ).reshape(3, 3);

        Assert.assertThat(a.broadcast(3, 3), valuesAndShapesMatch(expectedByRow));

        IntegerTensor expectedByColumn = IntegerTensor.create(
            1, 1, 1,
            2, 2, 2,
            3, 3, 3
        ).reshape(3, 3);

        Assert.assertThat(a.reshape(3, 1).broadcast(3, 3), valuesAndShapesMatch(expectedByColumn));
    }

    @Test
    public void canMod() {
        IntegerTensor value = IntegerTensor.create(4, 5);

        assertThat(value.mod(3), equalTo(IntegerTensor.create(1, 2)));
        assertThat(value.mod(2), equalTo(IntegerTensor.create(0, 1)));
        assertThat(value.mod(4), equalTo(IntegerTensor.create(0, 1)));
    }

    @Test
    public void canArgFindMaxOfOneByOne() {
        IntegerTensor tensor = IntegerTensor.scalar(1).reshape(1, 1);

        assertThat(tensor.argMax().scalar(), equalTo(0));
        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.create(0)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.create(0)));
    }

    @Test
    public void comparesIntegerScalarWithTensor() {
        IntegerTensor value = IntegerTensor.scalar(1);
        IntegerTensor differentValue = IntegerTensor.create(1, 2, 3);
        BooleanTensor result = value.elementwiseEquals(differentValue);
        assertThat(result, hasValue(true, false, false));
    }

    @Test
    public void canModScalar() {
        IntegerTensor value = IntegerTensor.scalar(4);

        assertThat(value.mod(3).scalar(), equalTo(1));
        assertThat(value.mod(2).scalar(), equalTo(0));
        assertThat(value.mod(4).scalar(), equalTo(0));
    }

    @Test
    public void doesKeepRankOnGTEq() {
        IntegerTensor value = IntegerTensor.scalar(1).reshape(1, 1, 1);
        assertEquals(3, value.greaterThanOrEqual(2).getRank());
    }

    @Test
    public void doesKeepRankOnGT() {
        IntegerTensor value = IntegerTensor.scalar(1).reshape(1, 1, 1);
        assertEquals(3, value.greaterThan(2).getRank());
    }

    @Test
    public void doesKeepRankOnLT() {
        IntegerTensor value = IntegerTensor.scalar(1).reshape(1, 1, 1);
        assertEquals(3, value.lessThan(2).getRank());
    }

    @Test
    public void doesKeepRankOnLTEq() {
        IntegerTensor value = IntegerTensor.scalar(1).reshape(1, 1, 1);
        assertEquals(3, value.lessThanOrEqual(2).getRank());
    }

    @Test
    public void canBroadcastScalarToShape() {
        IntegerTensor a = IntegerTensor.scalar(2);

        IntegerTensor expected = IntegerTensor.create(
            2, 2, 2,
            2, 2, 2,
            2, 2, 2
        ).reshape(3, 3);

        Assert.assertThat(a.broadcast(3, 3), valuesAndShapesMatch(expected));
    }

    @Test
    public void canBooleanIndex() {
        IntegerTensor a = IntegerTensor.scalar(1);
        IntegerTensor result = a.get(BooleanTensor.scalar(false));

        assertThat(result.getLength(), equalTo(0L));
    }

}

package io.improbable.keanu.tensor.dbl;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static junit.framework.TestCase.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import io.improbable.keanu.tensor.KeanuValueException;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;

public class Nd4jDoubleTensorTest {

    Nd4jDoubleTensor matrixA;
    Nd4jDoubleTensor matrixB;
    Nd4jDoubleTensor scalarA;
    Nd4jDoubleTensor vectorA;
    Nd4jDoubleTensor vectorB;
    Nd4jDoubleTensor rankThreeTensor;

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Before
    public void setup() {
        matrixA = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        matrixB = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        scalarA = Nd4jDoubleTensor.scalar(2.0);
        vectorA = Nd4jDoubleTensor.create(new double[]{1, 2, 3}, new int[]{3, 1});
        vectorB = Nd4jDoubleTensor.create(new double[]{1, 2, 3}, new int[]{1, 3});
        rankThreeTensor = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 2, 2});
    }

    @Test
    public void canElementWiseMultiplyMatrix() {
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
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{1, 2, -2, -2}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereGreaterThanAScalar() {
        DoubleTensor mask = matrixA.getGreaterThanMask(Nd4jDoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{1, 2, -2, -2}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanOrEqualAMatrix() {
        DoubleTensor mask = matrixA.getLessThanOrEqualToMask(Nd4jDoubleTensor.create(new double[]{2, 2, 2, 2}, new int[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, -2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanOrEqualAScalar() {
        DoubleTensor mask = matrixA.getLessThanOrEqualToMask(Nd4jDoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, -2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanAMatrix() {
        DoubleTensor mask = matrixA.getLessThanMask(Nd4jDoubleTensor.create(new double[]{2, 2, 2, 2}, new int[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanAScalar() {
        DoubleTensor mask = matrixA.getLessThanMask(Nd4jDoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    /**
     * Zero is a special case because it's usually the value that the mask uses to mean "false"
     */
    @Test

    public void canSetToZero() {
        DoubleTensor mask = matrixA.getLessThanMask(Nd4jDoubleTensor.create(new double[]{2, 2, 2, 2}, new int[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, 0.0);

        assertArrayEquals(new double[]{0, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canTestIfIsNaN() {
        Nd4jDoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1, 2, Double.NaN, 4}, new int[]{2, 2});
        assertThat(matrix.isNaN(), hasValue(false, false, true, false));
    }

    @Test
    public void canSetWhenNaN() {
        Nd4jDoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1, 2, Double.NaN, 4}, new int[]{2, 2});

        DoubleTensor mask = DoubleTensor.ones(matrix.getShape());
        DoubleTensor result = matrix.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, -2, -2, -2}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetToZeroWhenNaN() {
        Nd4jDoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1, 2, Double.NaN, 4}, new int[]{2, 2});

        DoubleTensor mask = DoubleTensor.ones(matrix.getShape());
        DoubleTensor result = matrix.setWithMaskInPlace(mask, 0.0);

        assertArrayEquals(new double[]{0, 0, 0, 0}, result.asFlatDoubleArray(), 0.0);
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
    public void canSuperBroadcast() {
        DoubleTensor x = Nd4jDoubleTensor.zeros(new int[]{2, 2, 2, 2});
        DoubleTensor y = Nd4jDoubleTensor.create(new double[]{1, 0, 1, 0}, new int[]{2, 2});

        DoubleTensor diff = x.plus(y);

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        }, new int[]{2, 2, 2, 2});

        assertEquals(expected, diff);
    }

    @Test
    public void canSuperBroadcastInPlace() {
        DoubleTensor x = Nd4jDoubleTensor.zeros(new int[]{2, 2, 2, 2});
        DoubleTensor y = Nd4jDoubleTensor.create(new double[]{1, 0, 1, 0}, new int[]{2, 2});

        DoubleTensor diff = x.plusInPlace(y);

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        }, new int[]{2, 2, 2, 2});

        assertEquals(expected, diff);
    }

    @Test
    public void canSetAllValues() {
        DoubleTensor rank5 = DoubleTensor.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, 7, 5, 8, 6,
            6, 3, 2, 9, 3, 4, 7, 6, 6, 2, 5, 4, 0, 2, 1, 3
        }, new int[]{2, 2, 2, 2, 2});
        rank5.setAllInPlace(0.0);
        assertAllValuesAre(rank5, 0.0);
        rank5.setAllInPlace(0.8);
        assertAllValuesAre(rank5, 0.8);
    }

    @Test
    public void canEqualsWithEpsilon() {
        double[] aData = new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, 7, 5, 8, 6,
            6, 3, 2, 9, 3, 4, 7, 6, 6, 2, 5, 4, 0, 2, 1, 3};
        double[] bData = new double[aData.length];
        for (int i = 0; i < aData.length; i++) {
            if (i % 2 == 0) {
                bData[i] = aData[i] + 0.4;
            } else {
                bData[i] = aData[i] - 0.4;
            }
        }
        double[] cData = bData.clone();
        cData[0] = cData[0] - 1.0;

        DoubleTensor a = DoubleTensor.create(aData, new int[]{2, 2, 2, 2, 2});
        DoubleTensor b = DoubleTensor.create(bData, new int[]{2, 2, 2, 2, 2});
        DoubleTensor c = DoubleTensor.create(cData, new int[]{2, 2, 2, 2, 2});
        assertTrue("equals with epsilon should be true", a.equalsWithinEpsilon(b, 0.5));
        assertTrue("equals with epsilon should be true (inverted order)", b.equalsWithinEpsilon(a, 0.5));
        assertTrue("equals with epsilon should be not true (max delta is 0.4)", !a.equalsWithinEpsilon(b, 0.2));
        assertTrue("equals with epsilon should be not true (max delta is 1.0)", !a.equalsWithinEpsilon(c, 0.5));
    }

    @Test
    public void doesClampTensor() {
        DoubleTensor A = Nd4jDoubleTensor.create(new double[]{0.25, 3, -4, -5}, new int[]{1, 4});
        DoubleTensor clampedA = A.clamp(DoubleTensor.scalar(-4.5), DoubleTensor.scalar(2.0));
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{0.25, 2.0, -4.0, -4.5}, new int[]{1, 4});
        assertEquals(expected, clampedA);
    }

    private void assertAllValuesAre(DoubleTensor tensor, double v) {
        for (double element : tensor.asFlatList()) {
            assertEquals(element, v, 0.01);
        }
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

    @Test
    public void canPermute() {
        DoubleTensor x = Nd4jDoubleTensor.create(new double[]{1, 2, 3}, new int[]{1, 3});
        DoubleTensor y = Nd4jDoubleTensor.create(new double[]{4, 5, 6}, new int[]{1, 3});

        DoubleTensor concatDimensionZero = DoubleTensor.concat(0, x, y);

        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, concatDimensionZero.asFlatDoubleArray(), 1e-6);

        DoubleTensor concatDimensionOne = DoubleTensor.concat(1, x, y);
        DoubleTensor permuttedConcatDimensionOne = concatDimensionOne.permute(1, 0);

        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, permuttedConcatDimensionOne.asFlatDoubleArray(), 1e-6);

        x = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 2, 2});
        y = Nd4jDoubleTensor.create(new double[]{9, 10, 11, 12, 13, 14, 15, 16}, new int[]{2, 2, 2});

        concatDimensionZero = DoubleTensor.concat(0, x, y);

        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, concatDimensionZero.asFlatDoubleArray(), 1e-6);

        concatDimensionOne = DoubleTensor.concat(1, x, y);
        permuttedConcatDimensionOne = concatDimensionOne.permute(1, 0, 2);

        double[] sliced = new double[permuttedConcatDimensionOne.asFlatDoubleArray().length / 2];
        for (int i = 0; i < permuttedConcatDimensionOne.asFlatDoubleArray().length / 2; i++) {
            sliced[i] = permuttedConcatDimensionOne.asFlatDoubleArray()[i];
        }

        DoubleTensor answer = DoubleTensor.create(sliced, x.getShape()).permute(1, 0, 2);
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, answer.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canLinSpace() {
        DoubleTensor actual = DoubleTensor.linspace(0, 10, 5);
        DoubleTensor expected = DoubleTensor.create(new double[]{0, 2.5, 5.0, 7.5, 10.0});
        assertEquals(expected, actual);
    }

    @Test
    public void canARange() {
        DoubleTensor actual = DoubleTensor.arange(0, 5);
        DoubleTensor expected = DoubleTensor.create(new double[]{0, 1, 2, 3, 4});
        assertEquals(expected, actual);
    }

    @Test
    public void canARangeWithStep() {
        DoubleTensor actual = DoubleTensor.arange(3, 7, 2);
        DoubleTensor expected = DoubleTensor.create(new double[]{3, 5});
        assertEquals(expected, actual);
    }

    @Test
    public void canARangeWithFractionStep() {
        DoubleTensor actual = DoubleTensor.arange(3, 7, 0.5);
        DoubleTensor expected = DoubleTensor.create(new double[]{3, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5});
        assertEquals(expected, actual);
    }

    @Test
    public void canARangeWithFractionStepThatIsNotEvenlyDivisible() {
        DoubleTensor actual = DoubleTensor.arange(3, 7, 1.5);
        DoubleTensor expected = DoubleTensor.create(new double[]{3.0, 4.5, 6.0});
        assertEquals(expected, actual);
    }

    @Test
    public void canTensorMultiplyWithVectorAndRank4() {
        DoubleTensor a = Nd4jDoubleTensor.create(new double[]{1, 2, 3}, new int[]{1, 1, 3, 1});
        DoubleTensor b = Nd4jDoubleTensor.create(new double[]{
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8
        }, new int[]{1, 3, 1, 5});

        DoubleTensor c = a.tensorMultiply(b, new int[]{2, 3}, new int[]{1, 0});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            30, 12, 18, 42, 48
        }, new int[]{1, 1, 1, 5});

        assertEquals(expected, c);
    }

    @Test
    public void canTensorMultiplyWithNumpyExample() {
        DoubleTensor a = DoubleTensor.arange(0, 60).reshape(3, 4, 5);
        DoubleTensor b = DoubleTensor.arange(0, 24.).reshape(4, 3, 2);
        DoubleTensor c = a.tensorMultiply(b, new int[]{1, 0}, new int[]{0, 1});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            4400., 4730.,
            4532., 4874.,
            4664., 5018.,
            4796., 5162.,
            4928., 5306.
        }, new int[]{5, 2});

        assertEquals(expected, c);
    }

    @Test
    public void canPermuteForTranspose() {
        DoubleTensor a = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        DoubleTensor permuted = a.permute(1, 0);
        DoubleTensor transposed = a.transpose();

        assertEquals(transposed, permuted);
    }

    @Test
    public void canPermuteUpperDimensions() {
        DoubleTensor a = DoubleTensor.create(new double[]{
            1, 2,
            3, 4,
            5, 6,
            7, 8
        }, new int[]{1, 2, 2, 2});
        DoubleTensor permuted = a.permute(0, 1, 3, 2);
        DoubleTensor expected = DoubleTensor.create(new double[]{
            1, 3,
            2, 4,
            5, 7,
            6, 8
        }, new int[]{1, 2, 2, 2});

        assertEquals(expected, permuted);
    }

    private void assertTimesOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        DoubleTensor actual = left.times(right);
        assertEquals(actual, expected);
    }

    private void assertTimesInPlaceOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        left = left.timesInPlace(right);
        assertEquals(left, expected);
    }

    private void assertPlusOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        DoubleTensor actual = left.plus(right);
        assertEquals(actual, expected);
    }

    private void assertPlusInPlaceOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        left = left.plusInPlace(right);
        assertEquals(left, expected);
    }

    private void assertDivideOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        DoubleTensor actual = left.div(right);
        assertEquals(actual, expected);
    }

    private void assertDivideInPlaceOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        left = left.divInPlace(right);
        assertEquals(left, expected);
    }

    private void assertMinusOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        DoubleTensor actual = left.minus(right);
        assertEquals(actual, expected);
    }

    private void assertMinusInPlaceOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        left = left.minusInPlace(right);
        assertEquals(left, expected);
    }

    @Test
    public void canCalculateProductOfVector() {
        double productVectorA = vectorA.product();
        double productVectorB = vectorB.product();
        double productRankThreeTensor = rankThreeTensor.product();

        assertEquals(6., productVectorA, 1e-6);
        assertEquals(6., productVectorB, 1e-6);
        assertEquals(40320, productRankThreeTensor, 1e-6);

        assertTrue(vectorA.isVector() && vectorB.isVector());
    }

    @Test
    public void scalarMinusInPlaceTensorBehavesSameAsMinus() {
        DoubleTensor scalar = DoubleTensor.scalar(1);
        DoubleTensor tensor = DoubleTensor.create(2, new int[]{1, 4});

        assertArrayEquals(scalar.minus(tensor).asFlatDoubleArray(), scalar.minusInPlace(tensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void scalarPlusInPlaceTensorBehavesSameAsPlus() {
        DoubleTensor scalar = DoubleTensor.scalar(1);
        DoubleTensor tensor = DoubleTensor.create(2, new int[]{1, 4});

        assertArrayEquals(scalar.plus(tensor).asFlatDoubleArray(), scalar.plusInPlace(tensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void scalarTimesInPlaceTensorBehavesSameAsTimes() {
        DoubleTensor scalar = DoubleTensor.scalar(1);
        DoubleTensor tensor = DoubleTensor.create(2, new int[]{1, 4});

        assertArrayEquals(scalar.times(tensor).asFlatDoubleArray(), scalar.timesInPlace(tensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void scalarDivInPlaceTensorBehavesSameAsDiv() {
        DoubleTensor scalar = DoubleTensor.scalar(1);
        DoubleTensor tensor = DoubleTensor.create(2, new int[]{1, 4});

        assertArrayEquals(scalar.div(tensor).asFlatDoubleArray(), scalar.divInPlace(tensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void smallerTensorMinusInPlaceLargerTensorBehavesSameAsMinus() {
        DoubleTensor smallerTensor = DoubleTensor.create(2, new int[]{2, 2});
        DoubleTensor largerTensor = DoubleTensor.create(3, new int[]{2, 2, 2});

        assertArrayEquals(smallerTensor.minus(largerTensor).asFlatDoubleArray(), smallerTensor.minusInPlace(largerTensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void smallerTensorPlusInPlaceLargerTensorBehavesSameAsPlus() {
        DoubleTensor smallerTensor = DoubleTensor.create(2, new int[]{2, 2});
        DoubleTensor largerTensor = DoubleTensor.create(3, new int[]{2, 2, 2});

        assertArrayEquals(smallerTensor.plus(largerTensor).asFlatDoubleArray(), smallerTensor.plusInPlace(largerTensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void smallerTensorTimesInPlaceLargerTensorBehavesSameAsTimes() {
        DoubleTensor smallerTensor = DoubleTensor.create(2, new int[]{2, 2});
        DoubleTensor largerTensor = DoubleTensor.create(3, new int[]{2, 2, 2});

        assertArrayEquals(smallerTensor.times(largerTensor).asFlatDoubleArray(), smallerTensor.timesInPlace(largerTensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void smallerTensorTimesInPlaceLargerTensorBehavesSameAsTimess() {
        DoubleTensor smallerTensor = DoubleTensor.create(2, new int[]{2, 2});
        DoubleTensor largerTensor = DoubleTensor.create(3, new int[]{2, 2, 2});

        assertArrayEquals(largerTensor.times(smallerTensor).asFlatDoubleArray(), largerTensor.timesInPlace(smallerTensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void smallerTensorDivInPlaceLargerTensorBehavesSameAsDiv() {
        DoubleTensor smallerTensor = DoubleTensor.create(2, new int[]{2, 2});
        DoubleTensor largerTensor = DoubleTensor.create(3, new int[]{2, 2, 2});

        assertArrayEquals(smallerTensor.div(largerTensor).asFlatDoubleArray(), smallerTensor.divInPlace(largerTensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalarTensor() {
        DoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1., 2., 3., 4.}, new int[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(Nd4jDoubleTensor.scalar(3.));
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }


    @Test
    public void canBroadcastMultiplyDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.ones(4, 2, 2, 2);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new int[]{4, 2, 2, 2});


        assertTimesOperationEquals(rank4, matrix, expected);
        assertTimesInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.ones(4, 2, 2, 2);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new int[]{4, 2, 2, 2});


        assertTimesOperationEquals(matrix, rank4, expected);
        assertTimesInPlaceOperationEquals(matrix, rank4, expected);
    }

    @Test
    public void canBroadcastPlusDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.zeros(new int[]{4, 2, 2, 2});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new int[]{4, 2, 2, 2});

        assertPlusOperationEquals(rank4, matrix, expected);
        assertPlusInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastPlusDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.zeros(new int[]{4, 2, 2, 2});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new int[]{4, 2, 2, 2});

        assertPlusOperationEquals(matrix, rank4, expected);
        assertPlusInPlaceOperationEquals(matrix, rank4, expected);
    }

    @Test
    public void canBroadcastDivideDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.ones(new int[]{4, 2, 2, 2}).times(10.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 5, 10}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
        }, new int[]{4, 2, 2, 2});

        assertDivideOperationEquals(rank4, matrix, expected);
        assertDivideInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastDivideDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.ones(new int[]{4, 2, 2, 2}).times(10.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 5, 10}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
        }, new int[]{4, 2, 2, 2});

        assertDivideOperationEquals(matrix, rank4, expected);
        assertDivideInPlaceOperationEquals(matrix, rank4, expected);
    }

    @Test
    public void canBroadcastMinusDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.ones(new int[]{4, 2, 2, 2}).times(5.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1
        }, new int[]{4, 2, 2, 2});

        assertMinusOperationEquals(rank4, matrix, expected);
        assertMinusInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMinusDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.ones(new int[]{4, 2, 2, 2}).times(5.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1
        }, new int[]{4, 2, 2, 2});

        assertMinusOperationEquals(matrix, rank4, expected);
        assertMinusInPlaceOperationEquals(matrix, rank4, expected);
    }

    @Test
    public void canSplit() {

        int dim = 2;
        DoubleTensor A = DoubleTensor.arange(0, 24).reshape(2, 3, 1, 4);
        DoubleTensor B = DoubleTensor.arange(24, 96).reshape(2, 3, 3, 4);
        DoubleTensor C = DoubleTensor.arange(96, 144).reshape(2, 3, 2, 4);

        DoubleTensor D = DoubleTensor.concat(dim, A, B, C);
        List<DoubleTensor> splitTensor = D.split(dim, new int[]{1, 4, 6});

        DoubleTensor[] concatList = new DoubleTensor[]{A, B, C};
        for (int i = 0; i < splitTensor.size(); i++) {
            assertEquals(concatList[i], splitTensor.get(i));
        }

    }

    @Test
    public void canSplitHighRank() {
        assertCanSplit(new int[]{2, 3, 4, 5, 7, 2}, new int[]{3, 2, 6}, 1);
    }

    @Test
    public void canSplitEndDimension() {
        assertCanSplit(new int[]{2, 3, 4, 5}, new int[]{3, 4, 2}, 3);
    }

    @Test
    public void canSplitFirstDimension() {
        assertCanSplit(new int[]{2, 3, 4, 5, 7, 2}, new int[]{3, 4, 2, 6, 9, 2}, 0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnZeroLengthSplit() {
        DoubleTensor A = DoubleTensor.arange(0, 100).reshape(10, 10);
        A.split(0, new int[]{0});
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnNegativeDimensionSplit() {
        DoubleTensor A = DoubleTensor.arange(0, 100).reshape(10, 10);
        A.split(-1, new int[]{1, 5});
    }

    @Test
    public void doesSatisfyJavaDocExample() {
        DoubleTensor A = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3}, 2, 6);

        List<DoubleTensor> actual = A.split(1, new int[]{1, 3, 6});

        DoubleTensor expected0 = DoubleTensor.create(new double[]{1, 7}, 1, 2);
        DoubleTensor expected1 = DoubleTensor.create(new double[]{2, 3, 8, 9}, 2, 2);
        DoubleTensor expected2 = DoubleTensor.create(new double[]{4, 5, 6, 1, 2, 3}, 2, 3);

        assertEquals(expected0, actual.get(0));
        assertEquals(expected1, actual.get(1));
        assertEquals(expected2, actual.get(2));
    }

    private void assertCanSplit(int[] baseShape, int[] concatenatedIndices, int concatenatedDimension) {

        int[] splitIndices = new int[concatenatedIndices.length];
        List<DoubleTensor> toConcat = new ArrayList<>();

        long previousEndLength = 0;
        int splitPosition = 0;
        for (int i = 0; i < concatenatedIndices.length; i++) {
            int[] shape = Arrays.copyOf(baseShape, baseShape.length);
            shape[concatenatedDimension] = concatenatedIndices[i];

            splitIndices[i] = splitPosition + concatenatedIndices[i];
            splitPosition = splitIndices[i];

            long newEndLength = previousEndLength + TensorShape.getLength(shape);
            toConcat.add(DoubleTensor.arange(previousEndLength, newEndLength).reshape(shape));
            previousEndLength = newEndLength;
        }

        DoubleTensor D = DoubleTensor.concat(concatenatedDimension, toConcat.toArray(new DoubleTensor[toConcat.size() - 1]));
        List<DoubleTensor> splitTensor = D.split(concatenatedDimension, splitIndices);

        for (int i = 0; i < splitTensor.size(); i++) {
            assertEquals(toConcat.get(i), splitTensor.get(i));
        }
    }

    @Test
    public void youCanCheckForZeros() {
        DoubleTensor containsZero = DoubleTensor.create(new double[]{
                0.0, -1.0, -Double.NEGATIVE_INFINITY, Double.NaN,
                Double.POSITIVE_INFINITY, Double.MIN_VALUE, Double.MAX_VALUE, -0.0},
            4, 2);

        BooleanTensor expectedMask = BooleanTensor.create(new boolean[]{
                false, true, true, true,
                true, true, true, false},
            4, 2);

        TensorValidator validator = TensorValidator.thatExpectsNotToFind(0.);
        assertThat(validator.check(containsZero), equalTo(expectedMask));
    }

    @Test
    public void youCanCheckForNaNs() {
        DoubleTensor containsNan = DoubleTensor.create(new double[]{
                0.0, -1.0, -Double.NEGATIVE_INFINITY, Double.NaN,
                Double.POSITIVE_INFINITY, Double.MIN_VALUE, Double.MAX_VALUE, -0.0},
            4, 2);

        BooleanTensor expectedMask = BooleanTensor.create(new boolean[]{
                true, true, true, false,
                true, true, true, true},
            4, 2);

        TensorValidator validator = TensorValidator.NAN_CATCHER;
        assertThat(validator.check(containsNan), equalTo(expectedMask));
        assertThat(containsNan.isNaN(), equalTo(expectedMask.not()));
    }

    @Test
    public void youCanReplaceNaNs() {
        double[] input = {
            0.0, -1.0, -Double.NEGATIVE_INFINITY, Double.NaN,
            Double.POSITIVE_INFINITY, Double.MIN_VALUE, Double.MAX_VALUE, -0.0};

        Double[] expectedOutput = {
            0.0, -1.0, -Double.NEGATIVE_INFINITY, 0.0,
            Double.POSITIVE_INFINITY, Double.MIN_VALUE, Double.MAX_VALUE, -0.0};

        DoubleTensor containsNan = DoubleTensor.create(input,
            4, 2);

        assertThat(containsNan.replaceNaN(0.), hasValue(expectedOutput));
    }

    @Test
    public void youCanDoYLogXEvenWhenBothAreZero() {
        DoubleTensor x = DoubleTensor.create(
            Double.MIN_VALUE, 1e-8, 1., 1e8);
        DoubleTensor y = x.duplicate();
        assertThat(x.log().times(y), equalTo(x.logTimes(y)));

        DoubleTensor zeros = DoubleTensor.create(0., -0.);

        assertThat(zeros.logTimes(zeros), hasValue(0., 0.));
        assertThat(zeros.log().times(zeros), hasValue(Double.NaN, Double.NaN));
    }

    @Test
    public void logTimesFailsIfYouPassInATensorThatAlreadyContainsNaN() {
        expectedException.expect(KeanuValueException.class);
        expectedException.expectMessage("Invalid value found");

        DoubleTensor x = DoubleTensor.create(1., 1.);
        DoubleTensor y = DoubleTensor.create(1., Double.NaN);
        x.logTimes(y);
    }

    @Test
    public void logTimesFailsIfYouStartWithATensorThatAlreadyContainsNaN() {
        expectedException.expect(KeanuValueException.class);
        expectedException.expectMessage("Invalid value found");

        DoubleTensor x = DoubleTensor.create(1., Double.NaN);
        DoubleTensor y = DoubleTensor.create(1., 1.);
        x.logTimes(y);
    }

    @Test
    public void youCanFixAValidationIssueByReplacingTheValue() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        DoubleTensor expectedResult = DoubleTensor.create(1.0, 1e-8, -1.0);

        TensorValidator validator = TensorValidator.thatReplaces(0., 1e-8);
        validator.validate(containsZero);
        assertThat(containsZero, equalTo(expectedResult));
    }

    @Test
    public void youCanFixACustomValidationIssueByReplacingTheValue() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        DoubleTensor expectedResult = DoubleTensor.create(1.0, 1e-8, 1e-8);

        TensorValidator<Double, Tensor<Double>> validator = TensorValidator.thatFixesElementwise(x -> x > 0., TensorValidationPolicy.changeValueTo(1e-8));
        validator.validate(containsZero);
        assertThat(containsZero, equalTo(expectedResult));
    }
}

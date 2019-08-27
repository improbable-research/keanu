package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensorFactory;
import io.improbable.keanu.tensor.lng.JVMLongTensorFactory;
import io.improbable.keanu.tensor.lng.LongTensor;
import junit.framework.TestCase;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.IntStream;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * This test class generically tests all listed Fixed Point Tensor types. To add another tensor implementation to be
 * tested, add a tensor factory and a function that converts a long (the largest fixed point data type) to the type
 * required by the tensor under test. In your test you can convert a long to the appropriate type and create a new
 * instance of the tensor implementation by using the factory.
 *
 * @param <NUMBER_TYPE>        The fixed point number type
 * @param <FIXED_POINT_TENSOR> The fixed point tensor implementation of the NUMBER_TYPE type
 */
@RunWith(Parameterized.class)
public class FixedPointTensorTest<NUMBER_TYPE extends Number, FIXED_POINT_TENSOR extends FixedPointTensor<NUMBER_TYPE, FIXED_POINT_TENSOR>> {

    @Parameterized.Parameters(name = "{index}: Test with {2}")
    public static Iterable<Object[]> data() {
        Function<Long, Long> toLong = in -> in;
        Function<Long, Integer> toInt = Long::intValue;

        return Arrays.asList(new Object[][]{
            {new JVMLongTensorFactory(), toLong, "JVM LongTensor"},
            {new Nd4jIntegerTensorFactory(), toInt, "Nd4jIntegerTensor"}
        });
    }

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    private FixedPointTensorFactory<NUMBER_TYPE, FIXED_POINT_TENSOR> factory;
    private Function<Long, NUMBER_TYPE> typed;

    public FixedPointTensorTest(FixedPointTensorFactory<NUMBER_TYPE, FIXED_POINT_TENSOR> factory, Function<Long, NUMBER_TYPE> typed, String name) {
        this.factory = factory;
        this.typed = typed;
    }

    private NUMBER_TYPE typed(long in) {
        return typed.apply(in);
    }

    @Test
    public void youCanCreateARankZeroTensor() {
        FIXED_POINT_TENSOR scalar = factory.scalar(2);
        assertEquals(typed(2), scalar.scalar());
        TestCase.assertEquals(0, scalar.getRank());
    }

    @Test
    public void youCanCreateARankOneTensor() {
        FIXED_POINT_TENSOR vector = factory.create(new int[]{1, 2, 3, 4, 5}, new long[]{5});
        assertEquals(typed(4), vector.getValue(3));
        TestCase.assertEquals(1, vector.getRank());
    }

    @Test
    public void doesMinusScalar() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.minus(typed(2));
        int[] expected = new int[]{-1, 0, 1, 2};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.minusInPlace(typed(2));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesPlusScalar() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.plus(typed(2));
        int[] expected = new int[]{3, 4, 5, 6};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.plusInPlace(typed(2));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesTimesScalar() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.times(typed(2));
        int[] expected = new int[]{2, 4, 6, 8};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.timesInPlace(typed(2));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesDivideScalar() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.div(typed(2));
        int[] expected = new int[]{0, 1, 1, 2};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.divInPlace(typed(2));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
        assertArrayEquals(new double[]{0.0, 1.0, 1.0, 2.0}, matrixA.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void doesElementwisePower() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR matrixB = factory.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.pow(matrixB);
        int[] expected = new int[]{1, 8, 9, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.powInPlace(matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesScalarPower() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.pow(typed(2));
        int[] expected = new int[]{1, 4, 9, 16};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.powInPlace(typed(2));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseMinus() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR matrixB = factory.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.minus(matrixB);
        int[] expected = new int[]{-1, -1, 1, 4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.minusInPlace(matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwisePlus() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR matrixB = factory.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.plus(matrixB);
        int[] expected = new int[]{3, 5, 5, 4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.plusInPlace(matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseTimes() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR matrixB = factory.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.times(matrixB);
        int[] expected = new int[]{2, 6, 6, 0};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.timesInPlace(matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseDivide() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR matrixC = factory.create(new int[]{5, -1, 7, 2}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.div(matrixC);
        int[] expected = new int[]{0, -2, 0, 2};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.divInPlace(matrixC);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
        assertArrayEquals(new double[]{0.0, -2.0, 0.0, 2.0}, matrixA.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void doesElementwiseUnaryMinus() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.unaryMinus();
        int[] expected = new int[]{-1, -2, -3, -4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.unaryMinusInPlace();
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseAbsolute() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{-1, -2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.abs();
        int[] expected = new int[]{1, 2, 3, 4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.absInPlace();
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseGreaterThanMask() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{-1, -2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR matrixTwos = factory.create(2, new long[]{2, 2});
        FIXED_POINT_TENSOR scalarTwo = factory.scalar(2);

        FIXED_POINT_TENSOR maskFromMatrix = matrix.greaterThanMask(matrixTwos);
        FIXED_POINT_TENSOR maskFromScalar = matrix.greaterThanMask(scalarTwo);

        int[] expected = new int[]{0, 0, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseGreaterThanOrEqualToMask() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR matrixTwos = factory.create(2, new long[]{2, 2});
        FIXED_POINT_TENSOR scalarTWo = factory.scalar(2);

        FIXED_POINT_TENSOR maskFromMatrix = matrix.greaterThanOrEqualToMask(matrixTwos);
        FIXED_POINT_TENSOR maskFromScalar = matrix.greaterThanOrEqualToMask(scalarTWo);

        int[] expected = new int[]{0, 1, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanMask() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR matrixTwos = factory.create(2, new long[]{2, 2});
        FIXED_POINT_TENSOR scalarTwo = factory.scalar(2);

        FIXED_POINT_TENSOR maskFromMatrix = matrix.lessThanMask(matrixTwos);
        FIXED_POINT_TENSOR maskFromScalar = matrix.lessThanMask(scalarTwo);

        int[] expected = new int[]{1, 0, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanOrEqualToMask() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR matrixTwos = factory.create(2, new long[]{2, 2});
        FIXED_POINT_TENSOR scalarTwo = factory.scalar(2);

        FIXED_POINT_TENSOR maskFromMatrix = matrix.lessThanOrEqualToMask(matrixTwos);
        FIXED_POINT_TENSOR maskFromScalar = matrix.lessThanOrEqualToMask(scalarTwo);

        int[] expected = new int[]{1, 1, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesSetWithMask() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR mask = factory.create(new int[]{1, 1, 0, 0}, new long[]{2, 2});
        FIXED_POINT_TENSOR expected = factory.create(new int[]{100, 100, 3, 4}, new long[]{2, 2});

        FIXED_POINT_TENSOR result = matrix.setWithMask(mask, typed(100));
        assertThat(result, valuesAndShapesMatch(expected));

        FIXED_POINT_TENSOR resultInPlace = matrix.setWithMaskInPlace(mask, typed(100));
        assertThat(resultInPlace, valuesAndShapesMatch(expected));
        assertThat(matrix, valuesAndShapesMatch(expected));
    }

    @Test
    public void canBroadcastSetIfMask() {
        FIXED_POINT_TENSOR tensor = factory.create(new int[]{1, 2, 3, 4}, 2, 2);
        FIXED_POINT_TENSOR mask = factory.scalar(1);
        assertThat(tensor.setWithMask(mask, typed(-2)), valuesAndShapesMatch(factory.create(-2, new long[]{2, 2})));
    }

    @Test
    public void cannotSetIfMaskLengthIsLargerThanTensorLength() {
        FIXED_POINT_TENSOR tensor = factory.scalar(3);
        FIXED_POINT_TENSOR mask = factory.create(new int[]{1, 1, 0, 1}, 2, 2);
        assertThat(tensor.setWithMask(mask, typed(-2)), valuesAndShapesMatch(factory.create(new int[]{-2, -2, 3, -2}, 2, 2)));
    }

    @Test
    public void doesApplyUnaryFunction() {
        FIXED_POINT_TENSOR matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrixA.apply(v -> typed(v.intValue() + 1));
        int[] expected = new int[]{2, 3, 4, 5};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FIXED_POINT_TENSOR resultInPlace = matrixA.applyInPlace(v -> typed(v.intValue() + 1));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesCompareLessThanScalar() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThan(typed(3));
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqualScalar() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual(typed(3));
        Boolean[] expected = new Boolean[]{true, true, true, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThan() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThan(otherMatrix);
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqual() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual(otherMatrix);
        Boolean[] expected = new Boolean[]{false, true, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThan() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThan(otherMatrix);
        Boolean[] expected = new Boolean[]{true, false, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqual() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(otherMatrix);
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanScalar() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThan(typed(3));
        Boolean[] expected = new Boolean[]{false, false, false, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalar() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(typed(3));
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalarTensor() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(factory.scalar(3));
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanMask() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrix.greaterThanMask(otherMatrix);
        int[] expected = new int[]{1, 0, 0, 0};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualMask() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrix.greaterThanOrEqualToMask(otherMatrix);
        int[] expected = new int[]{1, 1, 0, 0};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanScalarMask() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrix.greaterThanMask(factory.scalar(3));
        int[] expected = new int[]{0, 0, 0, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalarMask() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrix.greaterThanOrEqualToMask(factory.scalar(3));
        int[] expected = new int[]{0, 0, 1, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualTensorMask() {
        FIXED_POINT_TENSOR matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FIXED_POINT_TENSOR result = matrix.greaterThanOrEqualToMask(factory.create(0, 4));
        int[] expected = new int[]{1, 0, 1, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void canElementwiseEqualsAScalarValue() {
        int value = 42;
        int otherValue = 43;
        FIXED_POINT_TENSOR allTheSame = factory.create(value, new long[]{2, 3});
        FIXED_POINT_TENSOR notAllTheSame = allTheSame.duplicate();
        notAllTheSame.setValue(typed(otherValue), 1, 1);

        assertThat(allTheSame.elementwiseEquals(typed(value)).allTrue().scalar(), equalTo(true));
        assertThat(notAllTheSame.elementwiseEquals(typed(value)), hasValue(true, true, true, true, false, true));
    }

    @Test
    public void canMatrixMultiply() {
        FIXED_POINT_TENSOR leftFixedPoint = factory.arange(typed(0), typed(6)).reshape(2, 3);

        if (!(leftFixedPoint instanceof IntegerTensor)) {
            //matrix multiply is only supported for ints at the moment.
            return;
        }

        FIXED_POINT_TENSOR rightixedPoint = factory.arange(typed(6), typed(12)).reshape(3, 2);
        LongTensor actual = leftFixedPoint.matrixMultiply(rightixedPoint).toLong();

        DoubleTensor left = DoubleTensor.arange(6).reshape(2, 3);
        DoubleTensor right = DoubleTensor.arange(6, 12).reshape(3, 2);
        LongTensor expected = left.matrixMultiply(right).toLong();

        assertThat(actual, valuesAndShapesMatch(expected));
    }

    @Test
    public void canTensorMultiply() {
        FIXED_POINT_TENSOR leftFixedPoint = factory.arange(typed(0), typed(12)).reshape(2, 3, 2);

        if (!(leftFixedPoint instanceof IntegerTensor)) {
            //matrix multiply is only supported for ints at the moment.
            return;
        }

        FIXED_POINT_TENSOR rightixedPoint = factory.arange(typed(6), typed(18)).reshape(3, 2, 2);
        LongTensor actual = leftFixedPoint.tensorMultiply(rightixedPoint, new int[]{1}, new int[]{0}).toLong();

        DoubleTensor left = DoubleTensor.arange(12).reshape(2, 3, 2);
        DoubleTensor right = DoubleTensor.arange(6, 18).reshape(3, 2, 2);
        LongTensor expected = left.tensorMultiply(right, new int[]{1}, new int[]{0}).toLong();

        assertThat(actual, valuesAndShapesMatch(expected));
    }

    @Test
    public void canBroadcastAdd() {
        FIXED_POINT_TENSOR x = factory.create(new int[]{1, 2, 3}, new long[]{3, 1});
        FIXED_POINT_TENSOR s = factory.create(new int[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new long[]{3, 5});

        FIXED_POINT_TENSOR diff = s.plus(x);

        FIXED_POINT_TENSOR expected = factory.create(new int[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new long[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastSubtract() {
        FIXED_POINT_TENSOR x = factory.create(new int[]{-1, -2, -3}, new long[]{3, 1});
        FIXED_POINT_TENSOR s = factory.create(new int[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new long[]{3, 5});

        FIXED_POINT_TENSOR diff = s.minus(x);

        FIXED_POINT_TENSOR expected = factory.create(new int[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new long[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastDivide() {
        FIXED_POINT_TENSOR x = factory.create(new int[]{1, 2, 3}, new long[]{3, 1});
        FIXED_POINT_TENSOR s = factory.create(new int[]{
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8
        }, new long[]{3, 5});

        FIXED_POINT_TENSOR diff = s.div(x);

        FIXED_POINT_TENSOR expected = factory.create(new int[]{
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
        FIXED_POINT_TENSOR tensor = factory.create(
            new int[]{numerator, numerator, numerator, numerator},
            new long[]{2, 2}
        );
        FIXED_POINT_TENSOR result = tensor.div(typed(denominator));
        assertArrayEquals(new int[]{expected, expected, expected, expected}, result.asFlatIntegerArray());
    }

    @Test
    public void canFindScalarMinAndMax() {
        FIXED_POINT_TENSOR a = factory.create(5, 4, 3, 2).reshape(2, 2);
        NUMBER_TYPE min = a.min().scalar();
        NUMBER_TYPE max = a.max().scalar();
        assertEquals(typed(2), min);
        assertEquals(typed(5), max);
    }

    @Test
    public void canFindMinAndMaxFromScalarToTensor() {
        FIXED_POINT_TENSOR a = factory.create(5, 4, 3, 2).reshape(1, 4);
        FIXED_POINT_TENSOR b = factory.scalar(3);

        FIXED_POINT_TENSOR min = a.min(b);
        FIXED_POINT_TENSOR max = a.max(b);

        assertArrayEquals(new int[]{3, 3, 3, 2}, min.asFlatIntegerArray());
        assertArrayEquals(new int[]{5, 4, 3, 3}, max.asFlatIntegerArray());
    }

    @Test
    public void canFindElementWiseMinAndMax() {
        FIXED_POINT_TENSOR a = factory.create(1, 2, 3, 4).reshape(1, 4);
        FIXED_POINT_TENSOR b = factory.create(2, 3, 1, 4).reshape(1, 4);

        FIXED_POINT_TENSOR min = a.min(b);
        FIXED_POINT_TENSOR max = a.max(b);

        assertArrayEquals(new int[]{1, 2, 1, 4}, min.asFlatIntegerArray());
        assertArrayEquals(new int[]{2, 3, 3, 4}, max.asFlatIntegerArray());
    }

    @Test
    public void canFindArgMaxOfRowVector() {
        FIXED_POINT_TENSOR tensorRow = factory.create(1, 3, 4, 5, 2).reshape(1, 5);

        assertThat(tensorRow.argMax().scalar(), equalTo(3));
        assertThat(tensorRow.argMax(0), valuesAndShapesMatch(IntegerTensor.zeros(5)));
        assertThat(tensorRow.argMax(1), valuesAndShapesMatch(IntegerTensor.create(new int[]{3}, 1)));
    }

    @Test
    public void canFindArgMaxOfColumnVector() {
        FIXED_POINT_TENSOR tensorCol = factory.create(1, 3, 4, 5, 2).reshape(5, 1);

        assertThat(tensorCol.argMax().scalar(), equalTo(3));
        assertThat(tensorCol.argMax(0), valuesAndShapesMatch(IntegerTensor.create(new int[]{3}, 1)));
        assertThat(tensorCol.argMax(1), valuesAndShapesMatch(IntegerTensor.zeros(5)));
    }

    @Test
    public void canFindArgMaxOfMatrix() {
        FIXED_POINT_TENSOR tensor = factory.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);

        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.create(1, 0, 0, 0)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.create(2, 0)));
        assertThat(tensor.argMax().scalar(), equalTo(2));
    }

    @Test
    public void canFindArgMinOfMatrix() {
        FIXED_POINT_TENSOR tensor = factory.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);

        assertThat(tensor.argMin(0), valuesAndShapesMatch(IntegerTensor.create(0, 1, 1, 1)));
        assertThat(tensor.argMin(1), valuesAndShapesMatch(IntegerTensor.create(0, 1)));
        assertThat(tensor.argMin().scalar(), equalTo(0));
    }

    @Test
    public void canFindArgMaxOfHighRank() {
        FIXED_POINT_TENSOR tensor = factory.create(IntStream.range(0, 512).toArray()).reshape(2, 8, 4, 2, 4);

        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.ones(8, 4, 2, 4)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.create(7, new long[]{2, 4, 2, 4})));
        assertThat(tensor.argMax(2), valuesAndShapesMatch(IntegerTensor.create(3, new long[]{2, 8, 2, 4})));
        assertThat(tensor.argMax(3), valuesAndShapesMatch(IntegerTensor.ones(2, 8, 4, 4)));
        assertThat(tensor.argMax().scalar(), equalTo(511));
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHigh() {
        FIXED_POINT_TENSOR tensor = factory.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);
        tensor.argMax(2);
    }

    @Test
    public void comparesWithScalar() {
        FIXED_POINT_TENSOR value = factory.create(1, 2, 3);
        FIXED_POINT_TENSOR differentValue = factory.scalar(1);
        BooleanTensor result = value.elementwiseEquals(differentValue);
        assertThat(result, hasValue(true, false, false));
    }

    @Test
    public void canSliceRank3To2() {
        FIXED_POINT_TENSOR x = factory.create(1, 2, 3, 4, 1, 2, 3, 4).reshape(2, 2, 2);
        TensorTestHelper.doesDownRankOnSliceRank3To2(x);
    }

    @Test
    public void canSliceRank2To1() {
        FIXED_POINT_TENSOR x = factory.create(1, 2, 3, 4).reshape(2, 2);
        TensorTestHelper.doesDownRankOnSliceRank2To1(x);
    }

    @Test
    public void canSliceRank1ToScalar() {
        FIXED_POINT_TENSOR x = factory.create(1, 2, 3, 4).reshape(4);
        TensorTestHelper.doesDownRankOnSliceRank1ToScalar(x);
    }

    @Test
    public void canBroadcastToShape() {
        FIXED_POINT_TENSOR a = factory.create(
            1, 2, 3
        ).reshape(3);

        FIXED_POINT_TENSOR expectedByRow = factory.create(
            1, 2, 3,
            1, 2, 3,
            1, 2, 3
        ).reshape(3, 3);

        Assert.assertThat(a.broadcast(3, 3), valuesAndShapesMatch(expectedByRow));

        FIXED_POINT_TENSOR expectedByColumn = factory.create(
            1, 1, 1,
            2, 2, 2,
            3, 3, 3
        ).reshape(3, 3);

        Assert.assertThat(a.reshape(3, 1).broadcast(3, 3), valuesAndShapesMatch(expectedByColumn));
    }


    @Test
    public void canMod() {
        FIXED_POINT_TENSOR value = factory.create(4, 5);

        assertThat(value.mod(typed(3)), equalTo(factory.create(1, 2)));
        assertThat(value.mod(typed(2)), equalTo(factory.create(0, 1)));
        assertThat(value.mod(typed(4)), equalTo(factory.create(0, 1)));
    }
}

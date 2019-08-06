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


@RunWith(Parameterized.class)
public class FixedPointTensorTest<N extends Number, T extends FixedPointTensor<N, T>> {

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

    private FixedPointTensorFactory<N, T> factory;
    private Function<Long, N> typed;

    public FixedPointTensorTest(FixedPointTensorFactory<N, T> factory, Function<Long, N> typed, String name) {
        this.factory = factory;
        this.typed = typed;
    }

    private N typed(long in) {
        return typed.apply(in);
    }

    @Test
    public void youCanCreateARankZeroTensor() {
        FixedPointTensor<N, T> scalar = factory.scalar(2);
        assertEquals(typed(2), scalar.scalar());
        TestCase.assertEquals(0, scalar.getRank());
    }

    @Test
    public void youCanCreateARankOneTensor() {
        FixedPointTensor<N, T> vector = factory.create(new int[]{1, 2, 3, 4, 5}, new long[]{5});
        assertEquals(typed(4), vector.getValue(3));
        TestCase.assertEquals(1, vector.getRank());
    }

    @Test
    public void doesMinusScalar() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.minus(typed(2));
        int[] expected = new int[]{-1, 0, 1, 2};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.minusInPlace(typed(2));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesPlusScalar() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.plus(typed(2));
        int[] expected = new int[]{3, 4, 5, 6};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.plusInPlace(typed(2));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesTimesScalar() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.times(typed(2));
        int[] expected = new int[]{2, 4, 6, 8};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.timesInPlace(typed(2));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesDivideScalar() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.div(typed(2));
        int[] expected = new int[]{0, 1, 1, 2};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.divInPlace(typed(2));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
        assertArrayEquals(new double[]{0.0, 1.0, 1.0, 2.0}, matrixA.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void doesElementwisePower() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> matrixB = factory.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.pow((T) matrixB);
        int[] expected = new int[]{1, 8, 9, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.powInPlace((T) matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesScalarPower() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.pow(typed(2));
        int[] expected = new int[]{1, 4, 9, 16};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.powInPlace(typed(2));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseMinus() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> matrixB = factory.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.minus((T) matrixB);
        int[] expected = new int[]{-1, -1, 1, 4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.minusInPlace((T) matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwisePlus() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> matrixB = factory.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.plus((T) matrixB);
        int[] expected = new int[]{3, 5, 5, 4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.plusInPlace((T) matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseTimes() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> matrixB = factory.create(new int[]{2, 3, 2, 0}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.times((T) matrixB);
        int[] expected = new int[]{2, 6, 6, 0};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.timesInPlace((T) matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseDivide() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> matrixC = factory.create(new int[]{5, -1, 7, 2}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.div((T) matrixC);
        int[] expected = new int[]{0, -2, 0, 2};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.divInPlace((T) matrixC);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
        assertArrayEquals(new double[]{0.0, -2.0, 0.0, 2.0}, matrixA.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void doesElementwiseUnaryMinus() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.unaryMinus();
        int[] expected = new int[]{-1, -2, -3, -4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.unaryMinusInPlace();
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseAbsolute() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{-1, -2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.abs();
        int[] expected = new int[]{1, 2, 3, 4};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.absInPlace();
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseGreaterThanMask() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{-1, -2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> matrixTwos = factory.create(2, new long[]{2, 2});
        FixedPointTensor<N, T> scalarTwo = factory.scalar(2);

        FixedPointTensor<N, T> maskFromMatrix = matrix.greaterThanMask((T) matrixTwos);
        FixedPointTensor<N, T> maskFromScalar = matrix.greaterThanMask((T) scalarTwo);

        int[] expected = new int[]{0, 0, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseGreaterThanOrEqualToMask() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> matrixTwos = factory.create(2, new long[]{2, 2});
        FixedPointTensor<N, T> scalarTWo = factory.scalar(2);

        FixedPointTensor<N, T> maskFromMatrix = matrix.greaterThanOrEqualToMask((T) matrixTwos);
        FixedPointTensor<N, T> maskFromScalar = matrix.greaterThanOrEqualToMask((T) scalarTWo);

        int[] expected = new int[]{0, 1, 1, 1};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanMask() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> matrixTwos = factory.create(2, new long[]{2, 2});
        FixedPointTensor<N, T> scalarTwo = factory.scalar(2);

        FixedPointTensor<N, T> maskFromMatrix = matrix.lessThanMask((T) matrixTwos);
        FixedPointTensor<N, T> maskFromScalar = matrix.lessThanMask((T) scalarTwo);

        int[] expected = new int[]{1, 0, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesElementwiseLessThanOrEqualToMask() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> matrixTwos = factory.create(2, new long[]{2, 2});
        FixedPointTensor<N, T> scalarTwo = factory.scalar(2);

        FixedPointTensor<N, T> maskFromMatrix = matrix.lessThanOrEqualToMask((T) matrixTwos);
        FixedPointTensor<N, T> maskFromScalar = matrix.lessThanOrEqualToMask((T) scalarTwo);

        int[] expected = new int[]{1, 1, 0, 0};
        assertArrayEquals(expected, maskFromMatrix.asFlatIntegerArray());
        assertArrayEquals(expected, maskFromScalar.asFlatIntegerArray());
    }

    @Test
    public void doesSetWithMask() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{-1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> mask = factory.create(new int[]{1, 1, 0, 0}, new long[]{2, 2});
        FixedPointTensor<N, T> expected = factory.create(new int[]{100, 100, 3, 4}, new long[]{2, 2});

        FixedPointTensor<N, T> result = matrix.setWithMask((T) mask, typed(100));
        assertThat(result, valuesAndShapesMatch(expected));

        FixedPointTensor<N, T> resultInPlace = matrix.setWithMaskInPlace((T) mask, typed(100));
        assertThat(resultInPlace, valuesAndShapesMatch(expected));
        assertThat(matrix, valuesAndShapesMatch(expected));
    }

    @Test
    public void canBroadcastSetIfMask() {
        FixedPointTensor<N, T> tensor = factory.create(new int[]{1, 2, 3, 4}, 2, 2);
        FixedPointTensor<N, T> mask = factory.scalar(1);
        assertThat(tensor.setWithMask((T) mask, typed(-2)), valuesAndShapesMatch(factory.create(-2, new long[]{2, 2})));
    }

    @Test
    public void cannotSetIfMaskLengthIsLargerThanTensorLength() {
        FixedPointTensor<N, T> tensor = factory.scalar(3);
        FixedPointTensor<N, T> mask = factory.create(new int[]{1, 1, 0, 1}, 2, 2);
        assertThat(tensor.setWithMask((T) mask, typed(-2)), valuesAndShapesMatch(factory.create(new int[]{-2, -2, 3, -2}, 2, 2)));
    }

    @Test
    public void doesApplyUnaryFunction() {
        FixedPointTensor<N, T> matrixA = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrixA.apply(v -> typed(v.intValue() + 1));
        int[] expected = new int[]{2, 3, 4, 5};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));

        FixedPointTensor<N, T> resultInPlace = matrixA.applyInPlace(v -> typed(v.intValue() + 1));
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, matrixA.asFlatIntegerArray());
    }

    @Test
    public void doesCompareLessThanScalar() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThan(typed(3));
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqualScalar() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual(typed(3));
        Boolean[] expected = new Boolean[]{true, true, true, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThan() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThan((T) otherMatrix);
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareLessThanOrEqual() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.lessThanOrEqual((T) otherMatrix);
        Boolean[] expected = new Boolean[]{false, true, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThan() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThan((T) otherMatrix);
        Boolean[] expected = new Boolean[]{true, false, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqual() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual((T) otherMatrix);
        Boolean[] expected = new Boolean[]{true, true, false, false};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanScalar() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThan(typed(3));
        Boolean[] expected = new Boolean[]{false, false, false, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalar() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(typed(3));
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalarTensor() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(factory.scalar(3));
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void doesCompareGreaterThanMask() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrix.greaterThanMask((T) otherMatrix);
        int[] expected = new int[]{1, 0, 0, 0};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualMask() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> otherMatrix = factory.create(new int[]{0, 2, 4, 7}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrix.greaterThanOrEqualToMask((T) otherMatrix);
        int[] expected = new int[]{1, 1, 0, 0};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanScalarMask() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrix.greaterThanMask(factory.scalar(3));
        int[] expected = new int[]{0, 0, 0, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalarMask() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrix.greaterThanOrEqualToMask(factory.scalar(3));
        int[] expected = new int[]{0, 0, 1, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void doesCompareGreaterThanOrEqualTensorMask() {
        FixedPointTensor<N, T> matrix = factory.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        FixedPointTensor<N, T> result = matrix.greaterThanOrEqualToMask(factory.create(0, 4));
        int[] expected = new int[]{1, 0, 1, 1};
        assertArrayEquals(expected, result.asFlatIntegerArray());
    }

    @Test
    public void canElementwiseEqualsAScalarValue() {
        int value = 42;
        int otherValue = 43;
        FixedPointTensor<N, T> allTheSame = factory.create(value, new long[]{2, 3});
        FixedPointTensor<N, T> notAllTheSame = allTheSame.duplicate();
        notAllTheSame.setValue(typed(otherValue), 1, 1);

        assertThat(allTheSame.elementwiseEquals(typed(value)).allTrue().scalar(), equalTo(true));
        assertThat(notAllTheSame.elementwiseEquals(typed(value)), hasValue(true, true, true, true, false, true));
    }

    @Test
    public void canMatrixMultiply() {
        T leftFixedPoint = factory.arange(typed(0), typed(6)).reshape(2, 3);

        if (!(leftFixedPoint instanceof IntegerTensor)) {
            //matrix multiply is only supported for ints at the moment.
            return;
        }

        T rightixedPoint = factory.arange(typed(6), typed(12)).reshape(3, 2);
        LongTensor actual = leftFixedPoint.matrixMultiply(rightixedPoint).toLong();

        DoubleTensor left = DoubleTensor.arange(6).reshape(2, 3);
        DoubleTensor right = DoubleTensor.arange(6, 12).reshape(3, 2);
        LongTensor expected = left.matrixMultiply(right).toLong();

        assertThat(actual, valuesAndShapesMatch(expected));
    }

    @Test
    public void canTensorMultiply() {
        T leftFixedPoint = factory.arange(typed(0), typed(12)).reshape(2, 3, 2);

        if (!(leftFixedPoint instanceof IntegerTensor)) {
            //matrix multiply is only supported for ints at the moment.
            return;
        }

        T rightixedPoint = factory.arange(typed(6), typed(18)).reshape(3, 2, 2);
        LongTensor actual = leftFixedPoint.tensorMultiply(rightixedPoint, new int[]{1}, new int[]{0}).toLong();

        DoubleTensor left = DoubleTensor.arange(12).reshape(2, 3, 2);
        DoubleTensor right = DoubleTensor.arange(6, 18).reshape(3, 2, 2);
        LongTensor expected = left.tensorMultiply(right, new int[]{1}, new int[]{0}).toLong();

        assertThat(actual, valuesAndShapesMatch(expected));
    }

    @Test
    public void canBroadcastAdd() {
        T x = factory.create(new int[]{1, 2, 3}, new long[]{3, 1});
        T s = factory.create(new int[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new long[]{3, 5});

        FixedPointTensor<N, T> diff = s.plus(x);

        FixedPointTensor<N, T> expected = factory.create(new int[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new long[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastSubtract() {
        FixedPointTensor<N, T> x = factory.create(new int[]{-1, -2, -3}, new long[]{3, 1});
        FixedPointTensor<N, T> s = factory.create(new int[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new long[]{3, 5});

        FixedPointTensor<N, T> diff = s.minus((T) x);

        FixedPointTensor<N, T> expected = factory.create(new int[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new long[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastDivide() {
        FixedPointTensor<N, T> x = factory.create(new int[]{1, 2, 3}, new long[]{3, 1});
        FixedPointTensor<N, T> s = factory.create(new int[]{
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8
        }, new long[]{3, 5});

        FixedPointTensor<N, T> diff = s.div((T) x);

        FixedPointTensor<N, T> expected = factory.create(new int[]{
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
        FixedPointTensor<N, T> tensor = factory.create(
            new int[]{numerator, numerator, numerator, numerator},
            new long[]{2, 2}
        );
        FixedPointTensor<N, T> result = tensor.div(typed(denominator));
        assertArrayEquals(new int[]{expected, expected, expected, expected}, result.asFlatIntegerArray());
    }

    @Test
    public void canFindScalarMinAndMax() {
        FixedPointTensor<N, T> a = factory.create(5, 4, 3, 2).reshape(2, 2);
        N min = a.min().scalar();
        N max = a.max().scalar();
        assertEquals(typed(2), min);
        assertEquals(typed(5), max);
    }

    @Test
    public void canFindMinAndMaxFromScalarToTensor() {
        FixedPointTensor<N, T> a = factory.create(5, 4, 3, 2).reshape(1, 4);
        FixedPointTensor<N, T> b = factory.scalar(3);

        FixedPointTensor<N, T> min = a.min((T) b);
        FixedPointTensor<N, T> max = a.max((T) b);

        assertArrayEquals(new int[]{3, 3, 3, 2}, min.asFlatIntegerArray());
        assertArrayEquals(new int[]{5, 4, 3, 3}, max.asFlatIntegerArray());
    }

    @Test
    public void canFindElementWiseMinAndMax() {
        FixedPointTensor<N, T> a = factory.create(1, 2, 3, 4).reshape(1, 4);
        FixedPointTensor<N, T> b = factory.create(2, 3, 1, 4).reshape(1, 4);

        FixedPointTensor<N, T> min = a.min((T) b);
        FixedPointTensor<N, T> max = a.max((T) b);

        assertArrayEquals(new int[]{1, 2, 1, 4}, min.asFlatIntegerArray());
        assertArrayEquals(new int[]{2, 3, 3, 4}, max.asFlatIntegerArray());
    }

    @Test
    public void canFindArgMaxOfRowVector() {
        FixedPointTensor<N, T> tensorRow = factory.create(1, 3, 4, 5, 2).reshape(1, 5);

        assertThat(tensorRow.argMax().scalar(), equalTo(3));
        assertThat(tensorRow.argMax(0), valuesAndShapesMatch(IntegerTensor.zeros(5)));
        assertThat(tensorRow.argMax(1), valuesAndShapesMatch(IntegerTensor.create(new int[]{3}, 1)));
    }

    @Test
    public void canFindArgMaxOfColumnVector() {
        FixedPointTensor<N, T> tensorCol = factory.create(1, 3, 4, 5, 2).reshape(5, 1);

        assertThat(tensorCol.argMax().scalar(), equalTo(3));
        assertThat(tensorCol.argMax(0), valuesAndShapesMatch(IntegerTensor.create(new int[]{3}, 1)));
        assertThat(tensorCol.argMax(1), valuesAndShapesMatch(IntegerTensor.zeros(5)));
    }

    @Test
    public void canFindArgMaxOfMatrix() {
        FixedPointTensor<N, T> tensor = factory.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);

        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.create(1, 0, 0, 0)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.create(2, 0)));
        assertThat(tensor.argMax().scalar(), equalTo(2));
    }

    @Test
    public void canFindArgMinOfMatrix() {
        FixedPointTensor<N, T> tensor = factory.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);

        assertThat(tensor.argMin(0), valuesAndShapesMatch(IntegerTensor.create(0, 1, 1, 1)));
        assertThat(tensor.argMin(1), valuesAndShapesMatch(IntegerTensor.create(0, 1)));
        assertThat(tensor.argMin().scalar(), equalTo(0));
    }

    @Test
    public void canFindArgMaxOfHighRank() {
        FixedPointTensor<N, T> tensor = factory.create(IntStream.range(0, 512).toArray()).reshape(2, 8, 4, 2, 4);

        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.ones(8, 4, 2, 4)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.create(7, new long[]{2, 4, 2, 4})));
        assertThat(tensor.argMax(2), valuesAndShapesMatch(IntegerTensor.create(3, new long[]{2, 8, 2, 4})));
        assertThat(tensor.argMax(3), valuesAndShapesMatch(IntegerTensor.ones(2, 8, 4, 4)));
        assertThat(tensor.argMax().scalar(), equalTo(511));
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHigh() {
        FixedPointTensor<N, T> tensor = factory.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);
        tensor.argMax(2);
    }

    @Test
    public void comparesWithScalar() {
        FixedPointTensor<N, T> value = factory.create(1, 2, 3);
        FixedPointTensor<N, T> differentValue = factory.scalar(1);
        BooleanTensor result = value.elementwiseEquals((T) differentValue);
        assertThat(result, hasValue(true, false, false));
    }

    @Test
    public void canSliceRank3To2() {
        FixedPointTensor<N, T> x = factory.create(1, 2, 3, 4, 1, 2, 3, 4).reshape(2, 2, 2);
        TensorTestHelper.doesDownRankOnSliceRank3To2(x);
    }

    @Test
    public void canSliceRank2To1() {
        FixedPointTensor<N, T> x = factory.create(1, 2, 3, 4).reshape(2, 2);
        TensorTestHelper.doesDownRankOnSliceRank2To1(x);
    }

    @Test
    public void canSliceRank1ToScalar() {
        FixedPointTensor<N, T> x = factory.create(1, 2, 3, 4).reshape(4);
        TensorTestHelper.doesDownRankOnSliceRank1ToScalar(x);
    }

    @Test
    public void canBroadcastToShape() {
        FixedPointTensor<N, T> a = factory.create(
            1, 2, 3
        ).reshape(3);

        FixedPointTensor<N, T> expectedByRow = factory.create(
            1, 2, 3,
            1, 2, 3,
            1, 2, 3
        ).reshape(3, 3);

        Assert.assertThat(a.broadcast(3, 3), valuesAndShapesMatch(expectedByRow));

        FixedPointTensor<N, T> expectedByColumn = factory.create(
            1, 1, 1,
            2, 2, 2,
            3, 3, 3
        ).reshape(3, 3);

        Assert.assertThat(a.reshape(3, 1).broadcast(3, 3), valuesAndShapesMatch(expectedByColumn));
    }


    @Test
    public void canMod() {
        FixedPointTensor<N, T> value = factory.create(4, 5);

        assertThat(value.mod(typed(3)), equalTo(factory.create(1, 2)));
        assertThat(value.mod(typed(2)), equalTo(factory.create(0, 1)));
        assertThat(value.mod(typed(4)), equalTo(factory.create(0, 1)));
    }
}

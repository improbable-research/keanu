package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.TensorMatchers;
import io.improbable.keanu.tensor.TensorTestHelper;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
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

    private IntegerTensor matrixA;
    private IntegerTensor scalarA;

    public Nd4jIntegerTensorTest() {
        matrixA = IntegerTensor.create(new int[]{1, 2, 3, 4}, 2, 2);
        scalarA = IntegerTensor.scalar(2);
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
    public void canEye() {
        IntegerTensor expected = IntegerTensor.create(new int[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3);
        IntegerTensor actual = IntegerTensor.eye(3);

        assertEquals(expected, actual);
    }

    @Test
    public void canDiag() {
        IntegerTensor expected = IntegerTensor.create(new int[]{1, 0, 0, 0, 2, 0, 0, 0, 3}, 3, 3);
        IntegerTensor actual = IntegerTensor.create(1, 2, 3).diag();

        assertEquals(expected, actual);
    }

    @Test
    public void canMatrixMultiply() {
        IntegerTensor left = IntegerTensor.create(new int[]{
            1, 2, 3,
            4, 5, 6
        }, 2, 3);

        IntegerTensor right = IntegerTensor.create(new int[]{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        }, 3, 3);

        IntegerTensor result = left.matrixMultiply(right);

        IntegerTensor expected = IntegerTensor.create(new int[]{
            30, 36, 42,
            66, 81, 96
        }, 2, 3);

        assertEquals(expected, result);
    }

    @Test
    public void canMatrixMultiply2x2() {

        IntegerTensor left = IntegerTensor.create(new int[]{
            1, 2,
            3, 4
        }, 2, 2);

        IntegerTensor right = IntegerTensor.create(new int[]{
            5, 6,
            7, 8
        }, 2, 2);

        IntegerTensor result = left.matrixMultiply(right);

        IntegerTensor expected = IntegerTensor.create(new int[]{
            19, 22,
            43, 50
        }, 2, 2);

        assertEquals(expected, result);
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
    public void doesMatrixTimesScalar() {
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
    public void doesScalarTimesMatrix() {
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor two = Nd4jIntegerTensor.scalar(2);
        IntegerTensor result = two.times(matrixA);
        int[] expected = new int[]{2, 4, 6, 8};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatIntegerArray()));
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
    public void doesDivideScalarByMatrix() {
        IntegerTensor result = scalarA.div(matrixA);
        assertArrayEquals(new double[]{2, 1, 0, 0}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canTranspose() {
        IntegerTensor a = IntegerTensor.create(new int[]{1, 2, 3, 4}, 2, 2);
        IntegerTensor actual = a.transpose();
        IntegerTensor expected = IntegerTensor.create(new int[]{1, 3, 2, 4}, 2, 2);

        assertEquals(expected, actual);
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotTransposeVector() {
        IntegerTensor.create(1, 2, 3).transpose();
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnIncorrectShape() {
        IntegerTensor.create(new int[]{0, 1, 2, 3, 4, 5, 6}, 2, 3);
    }

    @Test
    public void canReshape() {
        IntegerTensor a = IntegerTensor.create(new int[]{0, 1, 2, 3, 4, 5}, 2, 3);
        IntegerTensor actual = a.reshape(3, 2);
        IntegerTensor expected = IntegerTensor.create(new int[]{0, 1, 2, 3, 4, 5}, 3, 2);

        assertEquals(actual, expected);
    }

    @Test
    public void canReshapeWithWildCardDim() {
        IntegerTensor a = IntegerTensor.create(new int[]{0, 1, 2, 3, 4, 5}, 2, 3);
        IntegerTensor expected = IntegerTensor.create(new int[]{0, 1, 2, 3, 4, 5}, 3, 2);

        assertEquals(a.reshape(3, -1), expected);
        assertEquals(a.reshape(-1, 2), expected);
    }

    @Test
    public void canReshapeWithWildCardDimEvenWithLengthOneDim() {
        IntegerTensor a = IntegerTensor.create(new int[]{0, 1, 2, 3, 4, 5}, 1, 6);
        IntegerTensor expected = IntegerTensor.create(new int[]{0, 1, 2, 3, 4, 5}, 1, 6);

        assertEquals(a.reshape(-1, 6), expected);
    }

    @Test
    public void scalarMinusInPlaceTensorBehavesSameAsMinus() {
        IntegerTensor scalar = IntegerTensor.scalar(1);
        IntegerTensor tensor = IntegerTensor.create(2, new long[]{1, 4});

        assertArrayEquals(scalar.minus(tensor).asFlatIntegerArray(), scalar.minusInPlace(tensor).asFlatIntegerArray());
    }

    @Test
    public void scalarPlusInPlaceTensorBehavesSameAsPlus() {
        IntegerTensor scalar = IntegerTensor.scalar(1);
        IntegerTensor tensor = IntegerTensor.create(2, new long[]{1, 4});

        assertArrayEquals(scalar.plus(tensor).asFlatIntegerArray(), scalar.plusInPlace(tensor).asFlatIntegerArray());
    }

    @Test
    public void scalarTimesInPlaceTensorBehavesSameAsTimes() {
        IntegerTensor scalar = IntegerTensor.scalar(1);
        IntegerTensor tensor = IntegerTensor.create(2, new long[]{1, 4});

        assertArrayEquals(scalar.times(tensor).asFlatIntegerArray(), scalar.timesInPlace(tensor).asFlatIntegerArray());
    }

    @Test
    public void scalarDivInPlaceTensorBehavesSameAsDiv() {
        IntegerTensor scalar = IntegerTensor.scalar(1);
        IntegerTensor tensor = IntegerTensor.create(2, new long[]{1, 4});

        assertArrayEquals(scalar.div(tensor).asFlatIntegerArray(), scalar.divInPlace(tensor).asFlatIntegerArray());
    }

    @Test
    public void smallerTensorMinusInPlaceLargerTensorBehavesSameAsMinus() {
        IntegerTensor smallerTensor = IntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor largerTensor = IntegerTensor.create(3, new long[]{2, 2, 2});

        assertArrayEquals(smallerTensor.minus(largerTensor).asFlatIntegerArray(), smallerTensor.minusInPlace(largerTensor).asFlatIntegerArray());
    }

    @Test
    public void smallerTensorPlusInPlaceLargerTensorBehavesSameAsPlus() {
        IntegerTensor smallerTensor = IntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor largerTensor = IntegerTensor.create(3, new long[]{2, 2, 2});

        assertArrayEquals(smallerTensor.plus(largerTensor).asFlatIntegerArray(), smallerTensor.plusInPlace(largerTensor).asFlatIntegerArray());
    }

    @Test
    public void smallerTensorTimesInPlaceLargerTensorBehavesSameAsTimes() {
        IntegerTensor smallerTensor = IntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor largerTensor = IntegerTensor.create(3, new long[]{2, 2, 2});

        assertArrayEquals(smallerTensor.times(largerTensor).asFlatIntegerArray(), smallerTensor.timesInPlace(largerTensor).asFlatIntegerArray());
    }

    @Test
    public void smallerTensorDivInPlaceLargerTensorBehavesSameAsDiv() {
        IntegerTensor smallerTensor = IntegerTensor.create(2, new long[]{2, 2});
        IntegerTensor largerTensor = IntegerTensor.create(3, new long[]{2, 2, 2});

        assertArrayEquals(smallerTensor.div(largerTensor).asFlatIntegerArray(), smallerTensor.divInPlace(largerTensor).asFlatIntegerArray());
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

    /**
     * Zero is a special case because it's usually the value that the mask uses to mean "false"
     */
    @Test
    public void canSetToZero() {
        IntegerTensor matrixA = Nd4jIntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});

        IntegerTensor mask = matrixA.getLessThanMask(IntegerTensor.create(new int[]{2, 2, 2, 2}, 2, 2));
        IntegerTensor result = matrixA.setWithMaskInPlace(mask, 0);

        assertArrayEquals(new double[]{0, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);

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
    public void doesApplyUnaryFunctionToScalar() {
        IntegerTensor scalarTwo = IntegerTensor.scalar(2);
        IntegerTensor result = scalarTwo.apply(a -> a * 2);
        assertEquals(4, result.scalar(), 0);

        IntegerTensor resultInPlace = scalarTwo.applyInPlace(a -> a * 2);
        assertEquals(4, resultInPlace.scalar(), 0);
        assertEquals(4, scalarTwo.scalar(), 0);
    }

    @Test
    public void doesApplyUnaryFunctionToMatrix() {
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
    public void doesApplyUnaryFunctionToRank3() {
        IntegerTensor rank3Tensor = IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);
        IntegerTensor result = rank3Tensor.apply(a -> a * 2);
        int[] expected = new int[]{2, 4, 6, 8, 10, 12, 14, 16};
        assertArrayEquals(expected, result.asFlatIntegerArray());
        assertFalse(Arrays.equals(expected, rank3Tensor.asFlatIntegerArray()));

        IntegerTensor resultInPlace = rank3Tensor.applyInPlace(a -> a * 2);
        assertArrayEquals(expected, resultInPlace.asFlatIntegerArray());
        assertArrayEquals(expected, rank3Tensor.asFlatIntegerArray());
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
    public void canTensorMultiplyWithVectorAndRank4() {
        IntegerTensor a = IntegerTensor.create(new int[]{1, 2, 3}, 1, 1, 3, 1);
        IntegerTensor b = IntegerTensor.create(new int[]{
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8
        }, 1, 3, 1, 5);

        IntegerTensor c = a.tensorMultiply(b, new int[]{2, 3}, new int[]{1, 0});

        IntegerTensor expected = IntegerTensor.create(new int[]{
            30, 12, 18, 42, 48
        }, 1, 1, 1, 5);

        assertEquals(expected, c);
    }

    @Test
    public void canTensorMultiplyWithNumpyExample() {
        IntegerTensor a = DoubleTensor.arange(0, 60).reshape(3, 4, 5).toInteger();
        IntegerTensor b = DoubleTensor.arange(0, 24.).reshape(4, 3, 2).toInteger();
        IntegerTensor c = a.tensorMultiply(b, new int[]{1, 0}, new int[]{0, 1});

        IntegerTensor expected = IntegerTensor.create(new int[]{
            4400, 4730,
            4532, 4874,
            4664, 5018,
            4796, 5162,
            4928, 5306
        }, 5, 2);

        assertEquals(expected, c);
    }

    @Test
    public void canTensorMultiplyAllDimensions() {
        IntegerTensor a = IntegerTensor.create(new int[]{2}).reshape(1);
        IntegerTensor b = IntegerTensor.create(new int[]{1, 2, 3, 4}).reshape(2, 1, 2);
        IntegerTensor resultAB = a.tensorMultiply(b, new int[]{0}, new int[]{1});
        IntegerTensor resultBA = b.tensorMultiply(a, new int[]{1}, new int[]{0});

        assertArrayEquals(new long[]{2, 2}, resultAB.getShape());
        assertArrayEquals(new long[]{2, 2}, resultBA.getShape());
    }

    @Test
    public void canPermuteUpperDimensions() {
        IntegerTensor a = IntegerTensor.create(new int[]{
            1, 2,
            3, 4,
            5, 6,
            7, 8
        }, 1, 2, 2, 2);
        IntegerTensor permuted = a.permute(0, 1, 3, 2);
        IntegerTensor expected = IntegerTensor.create(new int[]{
            1, 3,
            2, 4,
            5, 7,
            6, 8
        }, 1, 2, 2, 2);

        assertEquals(expected, permuted);
    }

    @Test
    public void canPermute() {
        IntegerTensor x = IntegerTensor.create(new int[]{1, 2, 3}, 1, 3);
        IntegerTensor y = IntegerTensor.create(new int[]{4, 5, 6}, 1, 3);

        IntegerTensor concatDimensionZero = IntegerTensor.concat(0, x, y);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6}, concatDimensionZero.asFlatIntegerArray());

        IntegerTensor concatDimensionOne = IntegerTensor.concat(1, x, y);
        IntegerTensor permuttedConcatDimensionOne = concatDimensionOne.permute(1, 0);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6}, permuttedConcatDimensionOne.asFlatIntegerArray());

        x = IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);
        y = IntegerTensor.create(new int[]{9, 10, 11, 12, 13, 14, 15, 16}, 2, 2, 2);

        concatDimensionZero = IntegerTensor.concat(0, x, y);

        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, concatDimensionZero.asFlatIntegerArray());

        concatDimensionOne = IntegerTensor.concat(1, x, y);
        permuttedConcatDimensionOne = concatDimensionOne.permute(1, 0, 2);

        int[] sliced = new int[permuttedConcatDimensionOne.asFlatIntegerArray().length / 2];
        for (int i = 0; i < permuttedConcatDimensionOne.asFlatDoubleArray().length / 2; i++) {
            sliced[i] = permuttedConcatDimensionOne.asFlatIntegerArray()[i];
        }

        IntegerTensor answer = IntegerTensor.create(sliced, x.getShape()).permute(1, 0, 2);
        assertArrayEquals(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, answer.asFlatIntegerArray());
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
        IntegerTensor b = IntegerTensor.scalar(3);

        IntegerTensor min = IntegerTensor.min(a, b);
        IntegerTensor max = IntegerTensor.max(a, b);

        assertArrayEquals(new int[]{3, 3, 3, 2}, min.asFlatIntegerArray());
        assertArrayEquals(new int[]{5, 4, 3, 3}, max.asFlatIntegerArray());
    }

    @Test
    public void canFindMinFromScalarToTensorInPlace() {
        IntegerTensor a = IntegerTensor.create(5, 4, 3, 2).reshape(1, 4);
        IntegerTensor b = IntegerTensor.scalar(3);

        a.minInPlace(b);

        assertArrayEquals(new int[]{3, 3, 3, 2}, a.asFlatIntegerArray());
    }

    @Test
    public void canFindMaxFromScalarToTensorInPlace() {
        IntegerTensor a = IntegerTensor.create(5, 4, 3, 2).reshape(1, 4);
        IntegerTensor b = IntegerTensor.scalar(3);

        a.maxInPlace(b);

        assertArrayEquals(new int[]{5, 4, 3, 3}, a.asFlatIntegerArray());
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
        assertThat(tensorRow.argMax(1), valuesAndShapesMatch(IntegerTensor.create(new int[]{3}, 1)));
    }

    @Test
    public void canFindArgMaxOfColumnVector() {
        IntegerTensor tensorCol = IntegerTensor.create(1, 3, 4, 5, 2).reshape(5, 1);

        assertEquals(3, tensorCol.argMax());
        assertThat(tensorCol.argMax(0), valuesAndShapesMatch(IntegerTensor.create(new int[]{3}, 1)));
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
        IntegerTensor differentValue = IntegerTensor.scalar(1);
        BooleanTensor result = value.elementwiseEquals(differentValue);
        assertThat(result, hasValue(true, false, false));
    }


    @Test
    public void canSumOverSpecifiedDimensionOfRank3() {
        IntegerTensor x = IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, new long[]{2, 2, 2});
        IntegerTensor summation = x.sum(2);
        IntegerTensor expected = IntegerTensor.create(new int[]{3, 7, 11, 15}, new long[]{2, 2});
        assertThat(summation, equalTo(expected));
        assertThat(summation.getShape(), equalTo(expected.getShape()));
    }

    @Test
    public void canSumOverSpecifiedDimensionOfMatrix() {
        IntegerTensor x = IntegerTensor.create(new int[]{1, 2, 3, 4}, new long[]{2, 2});
        IntegerTensor summationRow = x.sum(1);
        IntegerTensor expected = IntegerTensor.create(3, 7);
        assertThat(summationRow, equalTo(expected));
        assertThat(summationRow.getShape(), equalTo(expected.getShape()));
    }

    @Test
    public void canSumOverSpecifiedDimensionOfVector() {
        IntegerTensor x = IntegerTensor.create(1, 2, 3, 4);
        IntegerTensor summation = x.sum(0);
        IntegerTensor expected = IntegerTensor.scalar(10);
        assertThat(summation.asFlatArray(), equalTo(expected.asFlatArray()));
        assertThat(summation.getShape(), equalTo(expected.getShape()));
    }

    @Test
    public void canDuplicateRank1() {
        IntegerTensor x = IntegerTensor.create(1, 2);
        assertEquals(x, x.duplicate());
    }

    @Test
    public void canDuplicateRank0() {
        IntegerTensor x = IntegerTensor.scalar(1);
        assertEquals(x, x.duplicate());
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
    public void canSliceRank1() {
        IntegerTensor x = IntegerTensor.create(1, 2, 3, 4).reshape(4);
        IntegerTensor slice = x.slice(0, 1);
        assertArrayEquals(new int[]{2}, slice.asFlatIntegerArray());
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
    public void canGetValueByIndex() {
        IntegerTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2).toInteger();
        double value = A.getValue(1, 0, 1);
        assertEquals(5, value, 1e-10);
    }

    @Test
    public void canGetValueByFlatIndex() {
        IntegerTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2).toInteger();
        double value = A.getValue(7);
        assertEquals(7, value, 1e-10);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnGetInvalidIndex() {
        IntegerTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2).toInteger();
        double value = A.getValue(1, 0);
    }

    @Test
    public void canSetValueByFlatIndex() {
        IntegerTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2).toInteger();
        A.setValue(1, 6);
        assertEquals(1, A.getValue(6), 0);
    }

    @Test
    public void canSetValueByIndex() {
        IntegerTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2).toInteger();
        A.setValue(1, 1, 0, 1);
        assertEquals(1, A.getValue(1, 0, 1), 0);
    }

    @Test(expected = Exception.class)
    public void doesThrowOnSetByInvalidIndex() {
        IntegerTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2).toInteger();
        A.setValue(1, 1, 0);
    }


}
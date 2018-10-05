package io.improbable.keanu.tensor.bool;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;

import static junit.framework.TestCase.assertTrue;

import java.util.Arrays;

import static org.junit.Assert.assertThat;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.junit.rules.ExpectedException;

public class SimpleBooleanTensorTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    BooleanTensor matrixA;
    BooleanTensor matrixB;
    BooleanTensor matrixC;

    @Before
    public void setup() {

        matrixA = SimpleBooleanTensor.create(new boolean[]{true, false, true, false}, new int[]{2, 2});
        matrixB = SimpleBooleanTensor.create(new boolean[]{false, false, true, true}, new int[]{2, 2});
        matrixC = SimpleBooleanTensor.create(new boolean[]{true, true, true, false}, new int[]{2, 2});
    }

    @Test
    public void doesElementwiseAnd() {
        BooleanTensor result = matrixA.and(matrixB);
        Boolean[] expected = new Boolean[]{false, false, true, false};
        assertArrayEquals(expected, result.asFlatArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatArray()));

        BooleanTensor resultInPlace = matrixA.andInPlace(matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatArray());
        assertArrayEquals(expected, matrixA.asFlatArray());
    }

    @Test
    public void doesElementwiseOr() {
        BooleanTensor result = matrixA.or(matrixB);
        Boolean[] expected = new Boolean[]{true, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatArray()));

        BooleanTensor resultInPlace = matrixA.orInPlace(matrixB);
        assertArrayEquals(expected, resultInPlace.asFlatArray());
        assertArrayEquals(expected, matrixA.asFlatArray());
    }

    @Test
    public void doesElementwiseNot() {
        BooleanTensor result = matrixA.not();
        Boolean[] expected = new Boolean[]{false, true, false, true};
        assertArrayEquals(expected, result.asFlatArray());
        assertFalse(Arrays.equals(expected, matrixA.asFlatArray()));

        BooleanTensor resultInPlace = matrixA.notInPlace();
        assertArrayEquals(expected, resultInPlace.asFlatArray());
        assertArrayEquals(expected, matrixA.asFlatArray());
    }

    @Test
    public void doesSetDoubleIf() {
        DoubleTensor trueCase = DoubleTensor.create(new double[]{1.5, 2.0, 3.3, 4.65}, new int[]{2, 2});
        DoubleTensor falseCase = DoubleTensor.create(new double[]{5.1, 7.2, 11.4, 23.22}, new int[]{2, 2});

        DoubleTensor result = matrixA.setDoubleIf(trueCase, falseCase);
        assertArrayEquals(new double[]{1.5, 7.2, 3.3, 23.22}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void doesSetIntegerIf() {
        IntegerTensor trueCase = IntegerTensor.create(new int[]{1, 2, 3, 4}, new int[]{2, 2});
        IntegerTensor falseCase = IntegerTensor.create(new int[]{5, 7, 11, 23}, new int[]{2, 2});

        IntegerTensor result = matrixA.setIntegerIf(trueCase, falseCase);
        assertArrayEquals(new int[]{1, 7, 3, 23}, result.asFlatIntegerArray());
    }

    @Test
    public void doesSetBooleanIf() {
        BooleanTensor result = matrixA.setBooleanIf(matrixB, matrixC);
        assertArrayEquals(new Boolean[]{false, true, true, false}, result.asFlatArray());
    }

    enum Something {
        A, B, C, D
    }

    @Test
    public void doesSetGenericIf() {

        Tensor<Something> trueCase = new GenericTensor<>(
            new Something[]{Something.A, Something.B, Something.C, Something.D},
            new int[]{2, 2}
        );

        Tensor<Something> falseCase = new GenericTensor<>(
            new Something[]{Something.D, Something.C, Something.C, Something.A},
            new int[]{2, 2}
        );

        Tensor<Something> result = matrixA.setIf(trueCase, falseCase);
        assertArrayEquals(
            new Something[]{Something.A, Something.C, Something.C, Something.A},
            result.asFlatArray()
        );
    }

    @Test
    public void doesAllTrue() {
        assertFalse(new SimpleBooleanTensor(new boolean[]{false, true, false, false}, new int[]{2, 2}).allTrue());
        assertTrue(new SimpleBooleanTensor(new boolean[]{true, true, true, true}, new int[]{2, 2}).allTrue());
    }

    @Test
    public void doesAllFalse() {
        assertFalse(new SimpleBooleanTensor(new boolean[]{false, true, false, false}, new int[]{2, 2}).allFalse());
        assertTrue(new SimpleBooleanTensor(new boolean[]{false, false, false, false}, new int[]{2, 2}).allFalse());
    }

    @Test
    public void canGetRandomAccessValue() {
        assertTrue(matrixA.getValue(0, 0));
        assertFalse(matrixA.getValue(0, 1));
        assertTrue(matrixA.getValue(1, 0));
        assertFalse(matrixA.getValue(1, 1));
    }

    @Test
    public void canSetRandomAccessValue() {
        matrixA.setValue(false, 0, 0);
        matrixA.setValue(true, 0, 1);
        matrixA.setValue(false, 1, 0);
        matrixA.setValue(true, 1, 1);

        assertArrayEquals(new Boolean[]{false, true, false, true}, matrixA.asFlatArray());
    }

    @Test
    public void canConvertToDoubles() {
        assertArrayEquals(new double[]{1.0, 0.0, 1.0, 0.0}, matrixA.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canConvertToIntegers() {
        assertArrayEquals(new int[]{1, 0, 1, 0}, matrixA.asFlatIntegerArray());
    }

    @Test
    public void canReshape() {
        BooleanTensor reshaped = matrixA.reshape(4, 1);
        assertArrayEquals(reshaped.asFlatIntegerArray(), matrixA.asFlatIntegerArray());
        assertArrayEquals(new int[]{4, 1}, reshaped.getShape());
    }

    @Test
    public void cannotSetIfMaskLengthIsSmallerThanTensorLength() {
        DoubleTensor mask = DoubleTensor.scalar(1.);

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The lengths of the tensor and mask must match, but got tensor length: " + matrixA.getLength() + ", mask length: " + mask.getLength());

        matrixA.setWithMaskInPlace(mask, false);
    }

    @Test
    public void cannotSetIfMaskLengthIsLargerThanTensorLength() {
        BooleanTensor tensor = BooleanTensor.scalar(false);
        DoubleTensor mask = DoubleTensor.create(new double[] {1., 1., 1., 1.}, 2, 2);

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The lengths of the tensor and mask must match, but got tensor length: " + tensor.getLength() + ", mask length: " + mask.getLength());

        tensor.setWithMaskInPlace(mask, false);
    }

    @Test
    public void canSetWithMaskIfLengthOfMaskAndNonScalarTensorAreEqual() {
        DoubleTensor mask = DoubleTensor.create(new double[] {1., 1., 0., 0.}, 2, 2);
        BooleanTensor result = matrixA.setWithMask(mask, false);
        assertArrayEquals(new Boolean[] {false, false, true, false}, result.asFlatArray());
    }

    @Test
    public void canSetWithMaskInPlaceIfLengthOfMaskAndNonScalarTensorAreEqual() {
        DoubleTensor mask = DoubleTensor.create(new double[] {1., 1., 0., 0.}, 2, 2);
        matrixA.setWithMaskInPlace(mask, false);
        assertArrayEquals(new Boolean[] {false, false, true, false}, matrixA.asFlatArray());
    }

    @Test
    public void canSetWithMaskInPlaceIfMaskAndTensorAreScalars() {
        BooleanTensor scalarTensor = BooleanTensor.scalar(false);
        DoubleTensor mask = DoubleTensor.scalar(1.);
        scalarTensor.setWithMaskInPlace(mask, true);

        assertTrue(scalarTensor.isScalar());
        assertTrue(scalarTensor.allTrue());
    }
}

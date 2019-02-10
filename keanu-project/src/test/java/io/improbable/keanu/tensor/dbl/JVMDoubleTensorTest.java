package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.validate.TensorValidator;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class JVMDoubleTensorTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    JVMDoubleTensor matrixA;
    JVMDoubleTensor matrixB;
    JVMDoubleTensor scalarA;
    JVMDoubleTensor vectorA;
    JVMDoubleTensor rankThreeTensor;

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Before
    public void setup() {
        matrixA = JVMDoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        matrixB = JVMDoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        scalarA = JVMDoubleTensor.scalar(2.0);
        vectorA = JVMDoubleTensor.create(new double[]{1, 2, 3}, new long[]{3});
        rankThreeTensor = JVMDoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new long[]{2, 2, 2});
    }

    @Before
    public void enableDebugModeForNaNChecking() {
        TensorValidator.NAN_CATCHER.enable();
        TensorValidator.NAN_FIXER.enable();
    }

    @After
    public void disableDebugModeForNaNChecking() {
        TensorValidator.NAN_CATCHER.disable();
        TensorValidator.NAN_FIXER.disable();
    }

    @Test
    public void youCanCreateARankZeroTensor() {
        DoubleTensor scalar = DoubleTensor.create(new double[]{2.0}, new long[]{});
        DoubleTensor expected = DoubleTensor.scalar(2.0);
        assertEquals(expected, scalar);
        assertEquals(0, scalar.getRank());
    }

    @Test
    public void youCanCreateARankOneTensor() {
        DoubleTensor vector = DoubleTensor.create(new double[]{1, 2, 3, 4, 5}, new long[]{5});
        assertArrayEquals(new long[]{5}, vector.getShape());
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
        DoubleTensor mask = matrixA.getGreaterThanMask(JVMDoubleTensor.create(new double[]{2, 2, 2, 2}, new long[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{1, 2, -2, -2}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereGreaterThanAScalar() {
        DoubleTensor mask = matrixA.getGreaterThanMask(JVMDoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{1, 2, -2, -2}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanOrEqualAMatrix() {
        DoubleTensor mask = matrixA.getLessThanOrEqualToMask(JVMDoubleTensor.create(new double[]{2, 2, 2, 2}, new long[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, -2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanOrEqualAScalar() {
        DoubleTensor mask = matrixA.getLessThanOrEqualToMask(JVMDoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, -2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanAMatrix() {
        DoubleTensor mask = matrixA.getLessThanMask(JVMDoubleTensor.create(new double[]{2, 2, 2, 2}, new long[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanAScalar() {
        DoubleTensor mask = matrixA.getLessThanMask(JVMDoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    /**
     * Zero is a special case because it's usually the value that the mask uses to mean "false"
     */
    @Test

    public void canSetToZero() {
        DoubleTensor mask = matrixA.getLessThanMask(JVMDoubleTensor.create(new double[]{2, 2, 2, 2}, new long[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, 0.0);

        assertArrayEquals(new double[]{0, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canTestIfIsNaN() {
        JVMDoubleTensor matrix = JVMDoubleTensor.create(new double[]{1, 2, Double.NaN, 4}, new long[]{2, 2});
        assertThat(matrix.isNaN(), hasValue(false, false, true, false));
    }

    @Test
    public void canSetWhenNaN() {
        JVMDoubleTensor matrix = JVMDoubleTensor.create(new double[]{1, 2, Double.NaN, 4}, new long[]{2, 2});

        DoubleTensor mask = DoubleTensor.ones(matrix.getShape());
        DoubleTensor result = matrix.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, -2, -2, -2}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetToZeroWhenNaN() {
        JVMDoubleTensor matrix = JVMDoubleTensor.create(new double[]{1, 2, Double.NaN, 4}, new long[]{2, 2});

        DoubleTensor mask = DoubleTensor.ones(matrix.getShape());
        DoubleTensor result = matrix.setWithMaskInPlace(mask, 0.0);

        assertArrayEquals(new double[]{0, 0, 0, 0}, result.asFlatDoubleArray(), 0.0);
    }

}

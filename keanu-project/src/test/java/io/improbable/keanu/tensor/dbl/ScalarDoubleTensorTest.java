package io.improbable.keanu.tensor.dbl;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class ScalarDoubleTensorTest {

    @Test
    public void doesClampScalarWithinBounds() {
        DoubleTensor A = DoubleTensor.scalar(0.25);
        DoubleTensor clampedA = A.clamp(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0));
        double expected = 0.25;
        assertEquals(expected, clampedA.scalar(), 0.0);
    }

    @Test
    public void doesClampScalarGreaterThanBounds() {
        DoubleTensor A = DoubleTensor.scalar(5);
        DoubleTensor clampedA = A.clamp(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0));
        double expected = 1.0;
        assertEquals(expected, clampedA.scalar(), 0.0);
    }

    @Test
    public void doesClampScalarLessThanBounds() {
        DoubleTensor A = DoubleTensor.scalar(-2);
        DoubleTensor clampedA = A.clamp(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0));
        double expected = 0.0;
        assertEquals(expected, clampedA.scalar(), 0.0);
    }
}

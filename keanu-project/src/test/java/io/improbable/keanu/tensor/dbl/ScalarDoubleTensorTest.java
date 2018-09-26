package io.improbable.keanu.tensor.dbl;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

public class ScalarDoubleTensorTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

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

    @Test
    public void youCanSliceOnDimensionZeroOrOne() {
        double value = 42.;
        DoubleTensor tensor = DoubleTensor.scalar(value);
        assertThat(tensor.slice(0, 0), hasValue(value));
        assertThat(tensor.slice(1, 0), hasValue(value));
    }

    @Test
    public void youCantSliceOnDimensionGreaterThanOne() {
        expectedException.expect(IllegalStateException.class);
        expectedException.expectMessage("Slice is only valid for dimension == 0 or 1 and index == 0 in a scalar");
        DoubleTensor.scalar(42.).slice(2, 0);
    }

    @Test
    public void youCantSliceOnIndexGreaterThanZero() {
        expectedException.expect(IllegalStateException.class);
        expectedException.expectMessage("Slice is only valid for dimension == 0 or 1 and index == 0 in a scalar");
        DoubleTensor.scalar(42.).slice(0, 1);
    }
}

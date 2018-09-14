package io.improbable.keanu.tensor.dbl;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;

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

    @Test
    public void youCanCheckForZeros() {
        DoubleTensor zero = DoubleTensor.scalar(0.);
        DoubleTensor nonZero = DoubleTensor.scalar(1e-8);
        TensorValidator validator = TensorValidator.thatChecksFor(0.);
        assertThat(validator.check(zero), equalTo(BooleanTensor.scalar(false)));
        assertThat(validator.check(nonZero), equalTo(BooleanTensor.scalar(true)));
    }

    @Test
    public void youCanCheckForNans() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor notNan = DoubleTensor.scalar(Double.NEGATIVE_INFINITY);
        TensorValidator validator = TensorValidator.thatChecksFor(Double.NaN);
        assertThat(validator.check(nan), equalTo(BooleanTensor.scalar(false)));
        assertThat(validator.check(notNan), equalTo(BooleanTensor.scalar(true)));
    }

    @Test
    public void youCanFixAValidationIssueByReplacingTheValue() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor zero = DoubleTensor.scalar(0.);
        DoubleTensor notNan = DoubleTensor.scalar(Double.NEGATIVE_INFINITY);
        TensorValidator validator = TensorValidator.thatChecksFor(Double.NaN).withPolicy(TensorValidationPolicy.changeValueTo(0.));
        assertThat(validator.validate(nan), equalTo(zero));
        assertThat(validator.validate(notNan), equalTo(notNan));
    }

    @Test
    public void youCanFixACustomValidationIssueByReplacingTheValue() {
        DoubleTensor zero = DoubleTensor.scalar(0.);
        DoubleTensor one = DoubleTensor.scalar(1.);
        DoubleTensor notZero = DoubleTensor.scalar(1e-8);
        TensorValidator validator = TensorValidator.thatExpects(x -> x > 0.).withPolicy(TensorValidationPolicy.changeValueTo(1e-8));
        assertThat(validator.validate(zero), equalTo(notZero));
        assertThat(validator.validate(one), equalTo(one));
    }
}

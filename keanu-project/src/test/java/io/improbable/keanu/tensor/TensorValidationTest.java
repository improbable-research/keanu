package io.improbable.keanu.tensor;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

import org.junit.Test;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.validate.TensorValidationPolicy;

public class TensorValidationTest {
    @Test
    public void youCanCheckTheValueOfATensor() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        BooleanTensor expectedMask = BooleanTensor.create(new boolean[] {false, true, false});

        TensorValidator validator = new TensorValueEqualsValidator(0.).withPolicy(TensorValidationPolicy.changeValueTo(1e-8));
        assertThat(validator.check(containsZero), equalTo(expectedMask));
    }

    @Test
    public void youCanChangeTheValueOfATensor() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        DoubleTensor expectedResult = DoubleTensor.create(1.0, 1e-8, -1.0);

        TensorValidator validator = new TensorValueEqualsValidator(0.).withPolicy(TensorValidationPolicy.changeValueTo(1e-8));
        assertThat(validator.validate(containsZero), equalTo(expectedResult));
    }

    @Test(expected = KeanuValueException.class)
    public void byDefaultItThrowsIfValidationFails() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        TensorValidator validator = new TensorValueEqualsValidator(0.);
        validator.validate(containsZero);
    }
}

package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import org.junit.Test;

import java.util.function.Function;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

public class TensorValidationTest {
    @Test
    public void youCanCheckTheValueOfATensor() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        BooleanTensor expectedMask = BooleanTensor.create(true, false, true);

        assertThat(TensorValidator.ZERO_CATCHER.check(containsZero), equalTo(expectedMask));
    }

    @Test
    public void youCanChangeTheValueOfATensor() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        DoubleTensor expectedResult = DoubleTensor.create(1.0, 1e-8, -1.0);

        TensorValidator<Double, DoubleTensor> validator = TensorValidator.thatReplaces(0., 1e-8);
        DoubleTensor actual = validator.validate(containsZero);
        assertThat(actual, equalTo(expectedResult));
    }

    @Test
    public void youCanChangeTheValueOfAScalar() {
        DoubleTensor containsZero = DoubleTensor.create(0.0);
        DoubleTensor expectedResult = DoubleTensor.create(1e-8);

        TensorValidator<Double, DoubleTensor> validator = TensorValidator.thatReplaces(0., 1e-8);
        DoubleTensor actual = validator.validate(containsZero);
        assertThat(actual, equalTo(expectedResult));
    }

    @Test
    public void ifTheValueIsScalarButThe() {
        DoubleTensor containsZero = DoubleTensor.create(0.0);
        DoubleTensor expectedResult = DoubleTensor.create(1e-8);

        TensorValidator<Double, DoubleTensor> validator = TensorValidator.thatReplaces(0., 1e-8);
        DoubleTensor actual = validator.validate(containsZero);
        assertThat(actual, equalTo(expectedResult));
    }

    @Test(expected = TensorValueException.class)
    public void byDefaultItThrowsIfValidationFails() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        TensorValidator.ZERO_CATCHER.validate(containsZero);
    }

    @Test
    public void youCanDefineCustomValidators() {
        Function<Double, Boolean> checkFunction = v -> v > 0.2;
        TensorValidator validator = TensorValidator.thatExpectsElementwise(checkFunction);
        DoubleTensor input = DoubleTensor.create(1.0, 0.1, -1.0);
        BooleanTensor expectedResult = BooleanTensor.create(new boolean[]{true, false, false});
        try {
            validator.validate(input);
            throw new AssertionError("Expected it to throw TensorValueException");
        } catch (TensorValueException e) {
            assertThat(e.getResult(), equalTo(expectedResult));
        }
    }
}

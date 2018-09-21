package io.improbable.keanu.tensor;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

import java.util.function.Function;

import org.junit.Test;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;

public class TensorValidationTest {
    @Test
    public void youCanCheckTheValueOfATensor() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        BooleanTensor expectedMask = BooleanTensor.create(new boolean[] {true, false, true});

        assertThat(TensorValidator.ZERO_CATCHER.check(containsZero), equalTo(expectedMask));
    }

    @Test
    public void youCanChangeTheValueOfATensor() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        DoubleTensor expectedResult = DoubleTensor.create(1.0, 1e-8, -1.0);

        TensorValidator validator = TensorValidator.thatReplaces(0., 1e-8);
        validator.validate(containsZero);
        assertThat(containsZero, equalTo(expectedResult));
    }

    @Test
    public void youCanChangeTheValueOfAScalar() {
        DoubleTensor containsZero = DoubleTensor.create(0.0);
        DoubleTensor expectedResult = DoubleTensor.create(1e-8);

        TensorValidator<Double, DoubleTensor> validator = TensorValidator.thatReplaces(0., 1e-8);
        containsZero = validator.validate(containsZero);
        assertThat(containsZero, equalTo(expectedResult));
    }

    @Test
    public void ifTheValueIsScalarButThe() {
        DoubleTensor containsZero = DoubleTensor.create(0.0);
        DoubleTensor expectedResult = DoubleTensor.create(1e-8);

        TensorValidator<Double, DoubleTensor> validator = TensorValidator.thatReplaces(0., 1e-8);
        containsZero = validator.validate(containsZero);
        assertThat(containsZero, equalTo(expectedResult));
    }

    @Test(expected = KeanuValueException.class)
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
            throw new AssertionError("Expected it to throw KeanuValueException");
        } catch(KeanuValueException e) {
            assertThat(e.getResult(), equalTo(expectedResult));
        }
    }
}

package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.TensorValueException;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertEquals;

public class ScalarDoubleTensorTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();


    @Before
    public void enableDebugModeForNaNChecking() throws Exception {
        TensorValidator.NAN_CATCHER.enable();
        TensorValidator.NAN_FIXER.enable();
    }

    @After
    public void disableDebugModeForNaNChecking() throws Exception {
        TensorValidator.NAN_CATCHER.disable();
        TensorValidator.NAN_FIXER.disable();
    }

    @Test
    public void canElementwiseEqualsAScalarValue() {
        double value = 42.0;
        DoubleTensor tensor = DoubleTensor.create(value);

        assertThat(tensor.elementwiseEquals(value), hasValue(true));
        assertThat(tensor.elementwiseEquals(value + 0.1), hasValue(false));
    }

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
    public void canTestIfIsNaN() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor notNan = DoubleTensor.scalar(Double.NEGATIVE_INFINITY);
        assertThat(nan.isNaN(), hasValue(true));
        assertThat(notNan.isNaN(), hasValue(false));
    }

    @Test
    public void canSetWhenNaN() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor mask = DoubleTensor.scalar(1.);
        assertThat(nan.setWithMaskInPlace(mask, 2.), hasValue(2.));
    }

    @Test
    public void canSetToZeroWhenNaN() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor mask = DoubleTensor.scalar(1.);
        assertThat(nan.setWithMaskInPlace(mask, 0.), hasValue(0.));
    }

    @Test
    public void youCanCheckForZeros() {
        DoubleTensor zero = DoubleTensor.scalar(0.);
        DoubleTensor nonZero = DoubleTensor.scalar(1e-8);
        TensorValidator<Double, DoubleTensor> validator = TensorValidator.ZERO_CATCHER;
        assertThat(validator.check(zero), equalTo(BooleanTensor.scalar(false)));
        assertThat(validator.check(nonZero), equalTo(BooleanTensor.scalar(true)));
    }

    @Test
    public void youCanCheckForNans() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor notNan = DoubleTensor.scalar(Double.NEGATIVE_INFINITY);
        TensorValidator<Double, DoubleTensor> validator = TensorValidator.NAN_CATCHER;
        assertThat(nan.isNaN(), equalTo(BooleanTensor.scalar(true)));
        assertThat(notNan.isNaN(), equalTo(BooleanTensor.scalar(false)));
        assertThat(validator.check(nan), equalTo(BooleanTensor.scalar(false)));
        assertThat(validator.check(notNan), equalTo(BooleanTensor.scalar(true)));
    }

    @Test
    public void youCanReplaceNaNs() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor notNan = DoubleTensor.scalar(Double.NEGATIVE_INFINITY);
        assertThat(nan.replaceNaN(0.), hasValue(0.));
        assertThat(notNan.replaceNaN(0.), hasValue(Double.NEGATIVE_INFINITY));
    }

    @Test
    public void youCanDoYLogXEvenWhenBothAreZero() {
        DoubleTensor zero = DoubleTensor.scalar(0.);
        assertThat(zero.safeLogTimes(zero), hasValue(0.));
        assertThat(zero.log().times(zero), hasValue(Double.NaN));
    }

    @Test
    public void youCanDoTensorYLogXEvenWhenBothAreZero() {
        DoubleTensor zero = DoubleTensor.scalar(0.);
        DoubleTensor zeroTensor = DoubleTensor.create(0., 0., 0.);
        assertThat(zero.log().times(zeroTensor), hasValue(Double.NaN, Double.NaN, Double.NaN));
        assertThat(zero.safeLogTimes(zeroTensor), hasValue(0., 0., 0.));
    }


    @Test
    public void logTimesFailsIfYouPassInATensorThatAlreadyContainsNaN() {
        expectedException.expect(TensorValueException.class);
        expectedException.expectMessage("Invalid value found");

        DoubleTensor x = DoubleTensor.scalar(1.);
        DoubleTensor y = DoubleTensor.scalar(Double.NaN);
        x.safeLogTimes(y);
    }

    @Test
    public void logTimesFailsIfYouStartWithATensorThatAlreadyContainsNaN() {
        expectedException.expect(TensorValueException.class);
        expectedException.expectMessage("Invalid value found");

        DoubleTensor x = DoubleTensor.scalar(Double.NaN);
        DoubleTensor y = DoubleTensor.scalar(1.);
        x.safeLogTimes(y);
    }

    @Test
    public void youCanFixAValidationIssueByReplacingTheValue() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor zero = DoubleTensor.scalar(0.);
        DoubleTensor notNan = DoubleTensor.scalar(Double.NEGATIVE_INFINITY);
        TensorValidator<Double, DoubleTensor> validator = TensorValidator.thatReplaces(Double.NaN, 0.);
        nan = validator.validate(nan);
        notNan = validator.validate(notNan);
        assertThat(nan, equalTo(zero));
        assertThat(notNan, equalTo(notNan));
    }

    @Test
    public void youCanFixACustomValidationIssueByReplacingTheValue() {
        DoubleTensor tensor1 = DoubleTensor.scalar(0.);
        DoubleTensor tensor2 = DoubleTensor.scalar(1.);
        DoubleTensor one = DoubleTensor.scalar(1.);
        DoubleTensor notZero = DoubleTensor.scalar(1e-8);
        Function<Double, Boolean> checkFunction = x -> x > 0.;
        TensorValidator<Double, DoubleTensor> validator = TensorValidator.thatFixesElementwise(checkFunction, TensorValidationPolicy.changeValueTo(1e-8));
        tensor1 = validator.validate(tensor1);
        tensor2 = validator.validate(tensor2);
        assertThat(tensor1, equalTo(notZero));
        assertThat(tensor2, equalTo(one));
    }

    @Test
    public void canArgFindMaxOfScalar() {
        DoubleTensor tensor = DoubleTensor.scalar(1);

        assertEquals(0, tensor.argMax());
        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.scalar(0)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.scalar(0)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHigh() {
        DoubleTensor tensor = DoubleTensor.scalar(1);
        tensor.argMax(2);
    }
}

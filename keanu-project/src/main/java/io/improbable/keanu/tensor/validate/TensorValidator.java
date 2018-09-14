package io.improbable.keanu.tensor.validate;

import java.util.function.Function;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.validate.check.CustomTensorValueChecker;
import io.improbable.keanu.tensor.validate.check.TensorValueChecker;
import io.improbable.keanu.tensor.validate.check.TensorValueNotEqualsCheck;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;
import io.improbable.keanu.tensor.validate.policy.ThrowValueException;

public class TensorValidator implements TensorValueChecker {

    public static TensorValidator thatChecksFor(double v) {
        return new TensorValidator(new TensorValueNotEqualsCheck(v));
    }

    public static TensorValidator thatExpects(Function<Double, Boolean> checkFunction) {
        return new TensorValidator(new CustomTensorValueChecker(checkFunction));
    }

    private final TensorValueChecker valueChecker;
    private TensorValidationPolicy validationPolicy;

    private TensorValidator(TensorValueChecker valueChecker) {
        this(valueChecker, new ThrowValueException("Invalid value found"));
    }

    private TensorValidator(TensorValueChecker valueChecker, TensorValidationPolicy validationPolicy) {
        this.valueChecker = valueChecker;
        this.validationPolicy = validationPolicy;
    }

    @Override
    public BooleanTensor check(DoubleTensor tensor) {
        return valueChecker.check(tensor);
    }

    public DoubleTensor validate(DoubleTensor containsZero) {
        BooleanTensor result = check(containsZero);
        return validationPolicy.handle(containsZero, result);
    }

    public TensorValidator withPolicy(TensorValidationPolicy validationPolicy) {
        this.validationPolicy = validationPolicy;
        return this;
    }
}

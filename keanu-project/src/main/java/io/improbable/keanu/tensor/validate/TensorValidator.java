package io.improbable.keanu.tensor.validate;

import java.util.function.Function;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.validate.check.CustomTensorValueChecker;
import io.improbable.keanu.tensor.validate.check.TensorValueChecker;
import io.improbable.keanu.tensor.validate.check.TensorValueNotEqualsCheck;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;

public class TensorValidator<U, T extends Tensor<U>> implements TensorValueChecker<T> {

    public static <U, T extends Tensor<U>> TensorValidator<U, T> thatExpectsNotToFind(U v) {
        return new TensorValidator<U, T>(new TensorValueNotEqualsCheck(v));
    }

    public static <T extends Tensor<Double>> TensorValidator<Double, T> thatChecksForNaN() {
        Function<Double, Boolean> checkFunction = v -> !Double.isNaN(v);
        return thatExpects(checkFunction);
    }

    public static <U, T extends Tensor<U>> TensorValidator<U, T> thatExpects(Function<U, Boolean> checkFunction) {
        return new TensorValidator<U, T>(new CustomTensorValueChecker<U, T>(checkFunction));
    }

    private final TensorValueChecker valueChecker;
    private TensorValidationPolicy<T> validationPolicy;

    private TensorValidator(TensorValueChecker<T> valueChecker) {
        this(valueChecker, TensorValidationPolicy.throwMessage("Invalid value found"));
    }

    private TensorValidator(TensorValueChecker<T> valueChecker, TensorValidationPolicy<T> validationPolicy) {
        this.valueChecker = valueChecker;
        this.validationPolicy = validationPolicy;
    }

    @Override
    public BooleanTensor check(T tensor) {
        return valueChecker.check(tensor);
    }

    public void validate(T tensor) {
        BooleanTensor result = check(tensor);
        validationPolicy.handle(tensor, result);
    }

    public TensorValidator withPolicy(TensorValidationPolicy validationPolicy) {
        this.validationPolicy = validationPolicy;
        return this;
    }
}

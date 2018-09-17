package io.improbable.keanu.tensor.validate;

import java.util.function.Function;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.validate.check.CustomTensorValueChecker;
import io.improbable.keanu.tensor.validate.check.TensorValueChecker;
import io.improbable.keanu.tensor.validate.check.TensorValueNotEqualsCheck;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;

public class TensorValidator<DATATYPE, TENSOR extends Tensor<DATATYPE>> implements TensorValueChecker<TENSOR> {

    public static final TensorValidator<Double, Tensor<Double>> NAN_VALIDATOR =TensorValidator.thatChecksForNaN();

    public static <DATATYPE, TENSOR extends Tensor<DATATYPE>> TensorValidator<DATATYPE, TENSOR> thatExpectsNotToFind(DATATYPE v) {
        return new TensorValidator<DATATYPE, TENSOR>(new TensorValueNotEqualsCheck(v));
    }

    public static <TENSOR extends Tensor<Double>> TensorValidator<Double, TENSOR> thatChecksForNaN() {
        Function<Double, Boolean> checkFunction = v -> !Double.isNaN(v);
        return thatExpects(checkFunction);
    }

    public static <DATATYPE, TENSOR extends Tensor<DATATYPE>> TensorValidator<DATATYPE, TENSOR> thatExpects(Function<DATATYPE, Boolean> checkFunction) {
        return new TensorValidator<DATATYPE, TENSOR>(new CustomTensorValueChecker<DATATYPE, TENSOR>(checkFunction));
    }

    private final TensorValueChecker valueChecker;
    private TensorValidationPolicy<TENSOR> validationPolicy;

    private TensorValidator(TensorValueChecker<TENSOR> valueChecker) {
        this(valueChecker, TensorValidationPolicy.throwMessage("Invalid value found"));
    }

    private TensorValidator(TensorValueChecker<TENSOR> valueChecker, TensorValidationPolicy<TENSOR> validationPolicy) {
        this.valueChecker = valueChecker;
        this.validationPolicy = validationPolicy;
    }

    @Override
    public BooleanTensor check(TENSOR tensor) {
        return valueChecker.check(tensor);
    }

    public void validate(TENSOR tensor) {
        BooleanTensor result = check(tensor);
        validationPolicy.handle(tensor, result);
    }

    public TensorValidator withPolicy(TensorValidationPolicy validationPolicy) {
        this.validationPolicy = validationPolicy;
        return this;
    }
}

package io.improbable.keanu.tensor.validate;

import java.util.function.Function;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.validate.check.CustomElementwiseTensorValueChecker;
import io.improbable.keanu.tensor.validate.check.CustomTensorValueChecker;
import io.improbable.keanu.tensor.validate.check.TensorValueChecker;
import io.improbable.keanu.tensor.validate.check.TensorValueNotEqualsCheck;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;

public class TensorValidator<DATATYPE, TENSOR extends Tensor<DATATYPE>> implements TensorValueChecker<TENSOR> {

    public static final TensorValidator<Double, DoubleTensor> NAN_CATCHER = TensorValidator.thatExpects(t -> t.isNaN().not());
    public static final TensorValidator<Double, Tensor<Double>> NAN_FIXER = TensorValidator
        .<Double, Tensor<Double>>thatExpectsElementwise(v -> !Double.isNaN(v))
        .<Double, Tensor<Double>>withPolicy(TensorValidationPolicy.changeValueTo(0.));

    public static <DATATYPE, TENSOR extends Tensor<DATATYPE>> TensorValidator<DATATYPE, TENSOR> thatExpectsNotToFind(DATATYPE v) {
        return new TensorValidator<DATATYPE, TENSOR>(new TensorValueNotEqualsCheck(v));
    }

    public static <DATATYPE, TENSOR extends Tensor<DATATYPE>> TensorValidator<DATATYPE, TENSOR> thatExpects(Function<TENSOR, BooleanTensor> checkFunction) {
        return new TensorValidator<>(new CustomTensorValueChecker<TENSOR>(checkFunction));
    }

    public static <DATATYPE, TENSOR extends Tensor<DATATYPE>> TensorValidator<DATATYPE, TENSOR> thatExpectsElementwise(Function<DATATYPE, Boolean> checkFunction) {
        return new TensorValidator<>(new CustomElementwiseTensorValueChecker<>(checkFunction));
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

    public TensorValidator<DATATYPE, TENSOR> withPolicy(TensorValidationPolicy<TENSOR> validationPolicy) {
        this.validationPolicy = validationPolicy;
        return this;
    }
}

package io.improbable.keanu.tensor.validate;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.validate.check.CustomElementwiseTensorValueChecker;
import io.improbable.keanu.tensor.validate.check.CustomTensorValueChecker;
import io.improbable.keanu.tensor.validate.check.TensorValueChecker;
import io.improbable.keanu.tensor.validate.check.TensorValueNotEqualsCheck;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;

import java.util.function.Function;

public interface TensorValidator<DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> extends TensorValueChecker<DATATYPE, TENSOR> {

    TENSOR validate(TENSOR tensor);

    TensorValidator<Double, DoubleTensor> ZERO_CATCHER = TensorValidator.thatExpectsNotToFind(0.);
    DebugTensorValidator<Double, DoubleTensor> NAN_CATCHER = new DebugTensorValidator<>(TensorValidator.thatExpects((Function<DoubleTensor, BooleanTensor>) t -> t.notNaN()));
    DebugTensorValidator<Double, DoubleTensor> NAN_FIXER = new DebugTensorValidator<>(new NaNFixingTensorValidator(0.0));

    static <DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> TensorCheckAndRespondValidator<DATATYPE, TENSOR> thatExpectsNotToFind(DATATYPE v) {
        return new TensorCheckAndRespondValidator<>((TensorValueChecker<DATATYPE, TENSOR>) new TensorValueNotEqualsCheck<>(v));
    }

    static <DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> TensorValidator<DATATYPE, TENSOR> thatReplaces(DATATYPE oldValue, DATATYPE newValue) {
        return new TensorCheckAndRespondValidator<>((TensorValueChecker<DATATYPE, TENSOR>) new TensorValueNotEqualsCheck<>(oldValue), TensorValidationPolicy.changeValueTo(newValue));
    }

    static <DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> TensorCheckAndRespondValidator<DATATYPE, TENSOR> thatExpects(Function<TENSOR, BooleanTensor> checkFunction) {
        return new TensorCheckAndRespondValidator<>(new CustomTensorValueChecker<>(checkFunction));
    }

    static <DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> TensorCheckAndRespondValidator<DATATYPE, TENSOR> thatExpectsElementwise(Function<DATATYPE, Boolean> checkFunction) {
        return new TensorCheckAndRespondValidator<>((TensorValueChecker<DATATYPE, TENSOR>) new CustomElementwiseTensorValueChecker<>(checkFunction));
    }

    static <DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> TensorCheckAndRespondValidator<DATATYPE, TENSOR> thatFixesElementwise(Function<DATATYPE, Boolean> checkFunction, TensorValidationPolicy<DATATYPE, TENSOR> validationPolicy) {
        return new TensorCheckAndRespondValidator<>(new CustomElementwiseTensorValueChecker<>(checkFunction), validationPolicy);
    }
}

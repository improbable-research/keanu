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

public interface TensorValidator<DATATYPE, TENSOR extends Tensor<DATATYPE>>
        extends TensorValueChecker<TENSOR> {

    TENSOR validate(TENSOR tensor);

    TensorValidator<Double, DoubleTensor> ZERO_CATCHER = TensorValidator.thatExpectsNotToFind(0.);
    DebugTensorValidator<Double, DoubleTensor> NAN_CATCHER =
            new DebugTensorValidator<>(TensorValidator.thatExpects(t -> t.notNaN()));
    DebugTensorValidator<Double, DoubleTensor> NAN_FIXER =
            new DebugTensorValidator<>(new NaNFixingTensorValidator(0.0));

    static <DATATYPE, TENSOR extends Tensor<DATATYPE>>
            TensorCheckAndRespondValidator<DATATYPE, TENSOR> thatExpectsNotToFind(DATATYPE v) {
        return new TensorCheckAndRespondValidator<>(new TensorValueNotEqualsCheck<>(v));
    }

    static <DATATYPE, TENSOR extends Tensor<DATATYPE>>
            TensorValidator<DATATYPE, TENSOR> thatReplaces(DATATYPE oldValue, DATATYPE newValue) {
        return new TensorCheckAndRespondValidator<>(
                new TensorValueNotEqualsCheck<>(oldValue),
                TensorValidationPolicy.changeValueTo(newValue));
    }

    static <DATATYPE, TENSOR extends Tensor<DATATYPE>>
            TensorCheckAndRespondValidator<DATATYPE, TENSOR> thatExpects(
                    Function<TENSOR, BooleanTensor> checkFunction) {
        return new TensorCheckAndRespondValidator<>(
                new CustomTensorValueChecker<TENSOR>(checkFunction));
    }

    static <DATATYPE, TENSOR extends Tensor<DATATYPE>>
            TensorCheckAndRespondValidator<DATATYPE, TENSOR> thatExpectsElementwise(
                    Function<DATATYPE, Boolean> checkFunction) {
        return new TensorCheckAndRespondValidator<>(
                new CustomElementwiseTensorValueChecker<>(checkFunction));
    }

    static <DATATYPE, TENSOR extends Tensor<DATATYPE>>
            TensorCheckAndRespondValidator<DATATYPE, TENSOR> thatFixesElementwise(
                    Function<DATATYPE, Boolean> checkFunction,
                    TensorValidationPolicy<TENSOR> validationPolicy) {
        return new TensorCheckAndRespondValidator<>(
                new CustomElementwiseTensorValueChecker<>(checkFunction), validationPolicy);
    }
}

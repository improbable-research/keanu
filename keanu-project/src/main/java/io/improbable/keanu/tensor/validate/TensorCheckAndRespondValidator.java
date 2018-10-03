package io.improbable.keanu.tensor.validate;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.validate.check.TensorValueChecker;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;

public class TensorCheckAndRespondValidator<DATATYPE, TENSOR extends Tensor<DATATYPE>>
        implements TensorValidator<DATATYPE, TENSOR> {

    private final TensorValueChecker valueChecker;
    private final TensorValidationPolicy<TENSOR> validationPolicy;

    TensorCheckAndRespondValidator(TensorValueChecker<TENSOR> valueChecker) {
        this(valueChecker, TensorValidationPolicy.throwMessage("Invalid value found"));
    }

    TensorCheckAndRespondValidator(
            TensorValueChecker<TENSOR> valueChecker,
            TensorValidationPolicy<TENSOR> validationPolicy) {
        this.valueChecker = valueChecker;
        this.validationPolicy = validationPolicy;
    }

    @Override
    public BooleanTensor check(TENSOR tensor) {
        return valueChecker.check(tensor);
    }

    public TENSOR validate(TENSOR tensor) {
        BooleanTensor result = check(tensor);
        return validationPolicy.handle(tensor, result);
    }
}

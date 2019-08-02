package io.improbable.keanu.tensor.validate;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.validate.check.TensorValueChecker;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;

public class TensorCheckAndRespondValidator<DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> implements TensorValidator<DATATYPE, TENSOR> {

    private final TensorValueChecker<DATATYPE, TENSOR> valueChecker;
    private final TensorValidationPolicy<DATATYPE, TENSOR> validationPolicy;

    TensorCheckAndRespondValidator(TensorValueChecker<DATATYPE, TENSOR> valueChecker) {
        this(valueChecker, TensorValidationPolicy.throwMessage("Invalid value found"));
    }

    TensorCheckAndRespondValidator(TensorValueChecker<DATATYPE, TENSOR> valueChecker, TensorValidationPolicy<DATATYPE, TENSOR> validationPolicy) {
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

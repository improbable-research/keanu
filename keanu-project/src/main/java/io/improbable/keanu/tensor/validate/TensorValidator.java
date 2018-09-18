package io.improbable.keanu.tensor.validate;

import java.util.function.Function;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.validate.check.CustomElementwiseTensorValueChecker;
import io.improbable.keanu.tensor.validate.check.CustomTensorValueChecker;
import io.improbable.keanu.tensor.validate.check.TensorValueNotEqualsCheck;

public interface TensorValidator<DATATYPE, TENSOR extends Tensor<DATATYPE>> {

    BooleanTensor check(TENSOR tensor);

    void validate(TENSOR tensor);

    TensorValidator<Double, DoubleTensor> NAN_CATCHER = TensorValidator.thatExpects(t -> t.isNaN().not());
    TensorValidator<Double, DoubleTensor> NAN_FIXER = new NaNFixingTensorValidator(0.0);

    public static <DATATYPE, TENSOR extends Tensor<DATATYPE>> TensorCheckAndRespondValidator<DATATYPE, TENSOR> thatExpectsNotToFind(DATATYPE v) {
        return new TensorCheckAndRespondValidator<DATATYPE, TENSOR>(new TensorValueNotEqualsCheck(v));
    }

    public static <DATATYPE, TENSOR extends Tensor<DATATYPE>> TensorCheckAndRespondValidator<DATATYPE, TENSOR> thatExpects(Function<TENSOR, BooleanTensor> checkFunction) {
        return new TensorCheckAndRespondValidator<>(new CustomTensorValueChecker<TENSOR>(checkFunction));
    }

    public static <DATATYPE, TENSOR extends Tensor<DATATYPE>> TensorCheckAndRespondValidator<DATATYPE, TENSOR> thatExpectsElementwise(Function<DATATYPE, Boolean> checkFunction) {
        return new TensorCheckAndRespondValidator<>(new CustomElementwiseTensorValueChecker<>(checkFunction));
    }
}

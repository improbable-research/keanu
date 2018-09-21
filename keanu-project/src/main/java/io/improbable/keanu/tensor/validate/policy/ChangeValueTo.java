package io.improbable.keanu.tensor.validate.policy;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public class ChangeValueTo<DATATYPE, TENSOR extends Tensor<DATATYPE>> implements TensorValidationPolicy<TENSOR> {
    private final DATATYPE value;

    // package private - because it's created by the factory method TensorValidationPolicy.changeValueTo
    ChangeValueTo(DATATYPE value) {
        this.value = value;
    }

    @Override
    public TENSOR handle(TENSOR tensor, BooleanTensor result) {
        if (tensor instanceof DoubleTensor) {
            Double value = (Double) this.value;
            DoubleTensor mask = result.not().toDoubleMask();
            tensor = (TENSOR) ((DoubleTensor) tensor).setWithMask(mask, value);
        } else if (tensor instanceof IntegerTensor) {
            Integer value = (Integer) this.value;
            IntegerTensor mask = result.not().toIntegerMask();
            tensor = (TENSOR) ((IntegerTensor) tensor).setWithMask(mask, value);
        } else {
            throw new ClassCastException("Cannot handle tensor of type " + tensor.getClass().getSimpleName());
        }
        return tensor;
    }
}

package io.improbable.keanu.tensor.validate.policy;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public class ChangeValueTo<U, T extends Tensor<U>> implements TensorValidationPolicy<T> {
    private final U value;

    ChangeValueTo(U value) {
        this.value = value;
    }

    @Override
    public void handle(T tensor, BooleanTensor result) {
        if (tensor instanceof DoubleTensor) {
            Double value = (Double) this.value;
            DoubleTensor mask = result.not().toDoubleMask();
            ((DoubleTensor) tensor).setWithMaskInPlace(mask, value);
        } else if (tensor instanceof IntegerTensor) {
            Integer value = (Integer) this.value;
            IntegerTensor mask = result.not().toIntegerMask();
            ((IntegerTensor) tensor).setWithMaskInPlace(mask, value);
        } else {
            throw new ClassCastException("Cannot handle tensor of type " + tensor.getClass().getSimpleName());
        }
    }
}

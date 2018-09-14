package io.improbable.keanu.tensor.validate.check;

import java.util.function.Function;

import org.nd4j.linalg.util.ArrayUtil;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;

public class CustomTensorValueChecker<U, T extends Tensor<U>> implements TensorValueChecker<T> {

    private final Function<U, Boolean> checkFunction;
    private TensorValidationPolicy policy = TensorValidationPolicy.throwMessage("Invalid value found");

    public CustomTensorValueChecker(Function<U, Boolean> checkFunction) {
        this.checkFunction = checkFunction;
    }

    @Override
    public BooleanTensor check(T tensor) {
        int length = ArrayUtil.prod(tensor.getShape());
        boolean[] results = new boolean[length];
        Tensor.FlattenedView<U> flattenedView = tensor.getFlattenedView();
        for (int i = 0; i < length; i++) {
            results[i] = checkFunction.apply(flattenedView.get(i));
        }
        return BooleanTensor.create(results, tensor.getShape());
    }
}

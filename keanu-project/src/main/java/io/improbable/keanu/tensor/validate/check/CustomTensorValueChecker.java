package io.improbable.keanu.tensor.validate.check;

import java.util.function.Function;

import org.nd4j.linalg.util.ArrayUtil;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;
import io.improbable.keanu.tensor.validate.policy.ThrowValueException;

public class CustomTensorValueChecker implements TensorValueChecker {

    private final Function<Double, Boolean> checkFunction;
    private TensorValidationPolicy policy = new ThrowValueException("Invalid value found");

    public CustomTensorValueChecker(Function<Double, Boolean> checkFunction) {
        this.checkFunction = checkFunction;
    }

    @Override
    public BooleanTensor check(DoubleTensor tensor) {
        int length = ArrayUtil.prod(tensor.getShape());
        boolean[] results = new boolean[length];
        Tensor.FlattenedView<Double> flattenedView = tensor.getFlattenedView();
        for (int i = 0; i < length; i++) {
            results[i] = checkFunction.apply(flattenedView.get(i));
        }
        return BooleanTensor.create(results, tensor.getShape());
    }
}

package io.improbable.keanu.tensor;

import java.util.function.Function;

import org.nd4j.linalg.util.ArrayUtil;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class CustomTensorValidator implements TensorValidator {

    private final Function<Double, Boolean> checkFunction;

    public CustomTensorValidator(Function<Double, Boolean> checkFunction) {
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

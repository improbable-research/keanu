package io.improbable.keanu.tensor.validate.check;

import java.util.function.Function;

import org.nd4j.linalg.util.ArrayUtil;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public class CustomTensorValueChecker<DATATYPE, TENSOR extends Tensor<DATATYPE>> implements TensorValueChecker<TENSOR> {

    private final Function<DATATYPE, Boolean> checkFunction;

    public CustomTensorValueChecker(Function<DATATYPE, Boolean> checkFunction) {
        this.checkFunction = checkFunction;
    }

    @Override
    public BooleanTensor check(TENSOR tensor) {
        int length = ArrayUtil.prod(tensor.getShape());
        boolean[] results = new boolean[length];
        Tensor.FlattenedView<DATATYPE> flattenedView = tensor.getFlattenedView();
        for (int i = 0; i < length; i++) {
            results[i] = checkFunction.apply(flattenedView.get(i));
        }
        return BooleanTensor.create(results, tensor.getShape());
    }
}

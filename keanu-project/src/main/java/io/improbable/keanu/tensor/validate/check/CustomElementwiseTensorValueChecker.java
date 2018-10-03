package io.improbable.keanu.tensor.validate.check;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import java.util.function.Function;
import org.nd4j.linalg.util.ArrayUtil;

public class CustomElementwiseTensorValueChecker<DATATYPE, TENSOR extends Tensor<DATATYPE>>
    implements TensorValueChecker<TENSOR> {

  private final Function<DATATYPE, Boolean> checkFunction;

  public CustomElementwiseTensorValueChecker(Function<DATATYPE, Boolean> checkFunction) {
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

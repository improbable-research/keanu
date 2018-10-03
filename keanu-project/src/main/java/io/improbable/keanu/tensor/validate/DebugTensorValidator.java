package io.improbable.keanu.tensor.validate;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public class DebugTensorValidator<DATATYPE, TENSOR extends Tensor<DATATYPE>>
    implements TensorValidator<DATATYPE, TENSOR> {
  private final TensorValidator<DATATYPE, TENSOR> delegate;
  private boolean debugMode = false;

  public DebugTensorValidator(TensorValidator<DATATYPE, TENSOR> delegate) {
    this.delegate = delegate;
  }

  @Override
  public TENSOR validate(TENSOR tensor) {
    if (debugMode) {
      return delegate.validate(tensor);
    } else {
      return tensor;
    }
  }

  @Override
  public BooleanTensor check(TENSOR tensor) {
    if (debugMode) {
      return delegate.check(tensor);
    } else {
      return BooleanTensor.trues(tensor.getShape());
    }
  }

  public void enable() {
    debugMode = true;
  }

  public void disable() {
    debugMode = false;
  }
}

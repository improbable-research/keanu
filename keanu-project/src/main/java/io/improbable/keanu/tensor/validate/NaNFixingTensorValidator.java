package io.improbable.keanu.tensor.validate;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class NaNFixingTensorValidator implements TensorValidator<Double, DoubleTensor> {

  private final double replacementValue;

  public NaNFixingTensorValidator(double replacementValue) {
    this.replacementValue = replacementValue;
  }

  @Override
  public BooleanTensor check(DoubleTensor tensor) {
    return tensor.notNaN();
  }

  @Override
  public DoubleTensor validate(final DoubleTensor tensor) {
    return tensor.replaceNaN(replacementValue);
  }
}

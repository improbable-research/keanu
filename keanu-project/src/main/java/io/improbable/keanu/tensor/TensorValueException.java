package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.bool.BooleanTensor;

public class TensorValueException extends RuntimeException {
  private final BooleanTensor result;

  public TensorValueException(String message, BooleanTensor result) {
    super(message);
    this.result = result;
  }

  public BooleanTensor getResult() {
    return result;
  }
}

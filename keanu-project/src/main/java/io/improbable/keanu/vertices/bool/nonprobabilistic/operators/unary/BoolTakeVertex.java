package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;

public class BoolTakeVertex extends BoolUnaryOpVertex<BooleanTensor> {
  private final int[] index;

  /**
   * A vertex that extracts a scalar at a given index
   *
   * @param inputVertex the input vertex to extract from
   * @param index the index to extract at
   */
  public BoolTakeVertex(BoolVertex inputVertex, int... index) {
    super(Tensor.SCALAR_SHAPE, inputVertex);
    this.index = index;
    TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
  }

  @Override
  protected BooleanTensor op(BooleanTensor value) {
    return BooleanTensor.scalar(value.getValue(index));
  }
}

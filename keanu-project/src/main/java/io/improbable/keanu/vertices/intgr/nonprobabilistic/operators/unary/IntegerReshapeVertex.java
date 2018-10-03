package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerReshapeVertex extends IntegerUnaryOpVertex {
  public IntegerReshapeVertex(IntegerVertex inputVertex, int... proposedShape) {
    super(proposedShape, inputVertex);
  }

  @Override
  protected IntegerTensor op(IntegerTensor value) {
    return value.reshape(getShape());
  }
}

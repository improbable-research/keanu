package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import static io.improbable.keanu.tensor.TensorShape.shapeSlice;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerSliceVertex extends IntegerUnaryOpVertex {
  private final int dimension;
  private final int index;

  /**
   * Takes the slice along a given dimension and index of a vertex
   *
   * @param inputVertex the input vertex
   * @param dimension the dimension to extract along
   * @param index the index of extraction
   */
  public IntegerSliceVertex(IntegerVertex inputVertex, int dimension, int index) {
    super(shapeSlice(dimension, inputVertex.getShape()), inputVertex);
    this.dimension = dimension;
    this.index = index;
  }

  @Override
  protected IntegerTensor op(IntegerTensor value) {
    return value.slice(dimension, index);
  }
}

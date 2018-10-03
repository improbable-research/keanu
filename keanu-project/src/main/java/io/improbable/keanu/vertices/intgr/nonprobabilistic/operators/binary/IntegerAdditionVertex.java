package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerAdditionVertex extends IntegerBinaryOpVertex {

  /**
   * Adds one vertex to another
   *
   * @param a a vertex to add
   * @param b a vertex to add
   */
  public IntegerAdditionVertex(IntegerVertex a, IntegerVertex b) {
    super(a, b);
  }

  @Override
  protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
    return l.plus(r);
  }
}

package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerMultiplicationVertex extends IntegerBinaryOpVertex {

  /**
   * Multiplies one vertex by another
   *
   * @param a a vertex to be multiplied
   * @param b a vertex to be multiplied
   */
  public IntegerMultiplicationVertex(IntegerVertex a, IntegerVertex b) {
    super(a, b);
  }

  @Override
  protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
    return l.times(r);
  }
}

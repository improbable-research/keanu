package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerDivisionVertex extends IntegerBinaryOpVertex {

  /**
   * Divides one vertex by another
   *
   * @param a a vertex to be divided
   * @param b a vertex to divide by
   */
  public IntegerDivisionVertex(IntegerVertex a, IntegerVertex b) {
    super(a, b);
  }

  @Override
  protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
    return l.div(r);
  }
}

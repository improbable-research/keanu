package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class LessThanOrEqualVertex<A extends NumberTensor, B extends NumberTensor>
    extends BoolBinaryOpVertex<A, B> {

  public LessThanOrEqualVertex(Vertex<A> a, Vertex<B> b) {
    super(a, b);
  }

  @Override
  protected BooleanTensor op(A l, B r) {
    return l.lessThanOrEqual(r);
  }
}

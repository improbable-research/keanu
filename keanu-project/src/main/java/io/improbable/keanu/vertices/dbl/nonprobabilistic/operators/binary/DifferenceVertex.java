package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import java.util.HashMap;
import java.util.Map;

public class DifferenceVertex extends DoubleBinaryOpVertex {

  /**
   * Subtracts one vertex from another
   *
   * @param left the vertex that will be subtracted from
   * @param right the vertex to subtract
   */
  public DifferenceVertex(DoubleVertex left, DoubleVertex right) {
    super(left, right);
  }

  @Override
  protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
    return l.minus(r);
  }

  @Override
  public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(
      PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
    Map<Vertex, PartialDerivatives> partials = new HashMap<>();
    partials.put(left, derivativeOfOutputsWithRespectToSelf);
    partials.put(right, derivativeOfOutputsWithRespectToSelf.multiplyBy(-1.0));
    return partials;
  }

  @Override
  protected DualNumber dualOp(DualNumber l, DualNumber r) {
    return l.minus(r);
  }
}

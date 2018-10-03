package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import java.util.Map;

public class CeilVertex extends DoubleUnaryOpVertex {

  /**
   * Applies the Ceiling operator to a vertex. This maps a vertex to the smallest integer greater
   * than or equal to its value
   *
   * @param inputVertex the vertex to be ceil'd
   */
  public CeilVertex(DoubleVertex inputVertex) {
    super(inputVertex);
  }

  @Override
  protected DoubleTensor op(DoubleTensor value) {
    return value.ceil();
  }

  @Override
  protected DualNumber dualOp(DualNumber dualNumber) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(
      PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
    throw new UnsupportedOperationException();
  }
}

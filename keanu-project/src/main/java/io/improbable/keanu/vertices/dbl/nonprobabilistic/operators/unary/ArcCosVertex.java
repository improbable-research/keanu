package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import java.util.HashMap;
import java.util.Map;

public class ArcCosVertex extends DoubleUnaryOpVertex {

  /**
   * Takes the inverse cosine of a vertex, Arccos(vertex)
   *
   * @param inputVertex the vertex
   */
  public ArcCosVertex(DoubleVertex inputVertex) {
    super(inputVertex);
  }

  @Override
  protected DoubleTensor op(DoubleTensor value) {
    return value.acos();
  }

  @Override
  protected DualNumber dualOp(DualNumber dualNumber) {
    return dualNumber.acos();
  }

  public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(
      PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
    DoubleTensor inputValue = inputVertex.getValue();

    // dArcCosdx = -1 / sqrt(1 - x^2)
    DoubleTensor dSelfWrtInput =
        inputValue
            .pow(2)
            .unaryMinusInPlace()
            .plusInPlace(1)
            .sqrtInPlace()
            .reciprocalInPlace()
            .unaryMinusInPlace();

    Map<Vertex, PartialDerivatives> partials = new HashMap<>();
    partials.put(
        inputVertex,
        derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(
            dSelfWrtInput, this.getShape()));

    return partials;
  }
}

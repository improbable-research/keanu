package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import java.util.HashMap;
import java.util.Map;

public class ArcSinVertex extends DoubleUnaryOpVertex {

  /**
   * Takes the inverse sin of a vertex, Arcsin(vertex)
   *
   * @param inputVertex the vertex
   */
  public ArcSinVertex(DoubleVertex inputVertex) {
    super(inputVertex);
  }

  @Override
  protected DoubleTensor op(DoubleTensor value) {
    return value.asin();
  }

  @Override
  protected DualNumber dualOp(DualNumber dualNumber) {
    return dualNumber.asin();
  }

  @Override
  public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(
      PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
    DoubleTensor inputValue = inputVertex.getValue();

    // dArcSindx = 1 / sqrt(1 - x^2)
    DoubleTensor dSelfWrtInput =
        inputValue.pow(2).unaryMinusInPlace().plusInPlace(1).sqrtInPlace().reciprocalInPlace();

    Map<Vertex, PartialDerivatives> partials = new HashMap<>();
    partials.put(
        inputVertex,
        derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(
            dSelfWrtInput, this.getShape()));

    return partials;
  }
}

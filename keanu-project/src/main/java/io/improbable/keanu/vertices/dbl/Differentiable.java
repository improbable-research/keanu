package io.improbable.keanu.vertices.dbl;

import java.util.Map;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public interface Differentiable {

    PartialDerivatives calculateDualNumber(Map<Vertex, PartialDerivatives> derivativeOfSelfWithRespectToInputs);

    Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf);

    default PartialDerivatives getDualNumber() {
        return Differentiator.calculateDual((Vertex & Differentiable) this);
    }
}

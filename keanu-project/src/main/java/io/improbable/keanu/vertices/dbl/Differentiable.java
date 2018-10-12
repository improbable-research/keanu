package io.improbable.keanu.vertices.dbl;

import java.util.Map;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public interface Differentiable {

    PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs);

    Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf);

    default PartialDerivatives getDerivativeWrtLatents() {
        return Differentiator.forwardModeAutoDiff((Vertex & Differentiable) this);
    }
}

package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import java.util.Map;

public interface Differentiable {

    DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers);

    Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(
            PartialDerivatives derivativeOfOutputsWithRespectToSelf);

    default DualNumber getDualNumber() {
        return Differentiator.calculateDual((Vertex & Differentiable) this);
    }
}

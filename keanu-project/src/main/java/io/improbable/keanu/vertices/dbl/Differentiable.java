package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Collections;
import java.util.Map;

public interface Differentiable {

    default PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInputs) {
        if (((Vertex) this).isObserved()) {
            return PartialDerivative.EMPTY;
        } else {
            return PartialDerivative.withRespectToSelf(((Vertex) this).getId(), ((Vertex) this).getShape());
        }
    }

    default Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputsWithRespectToSelf) {
        return Collections.emptyMap();
    }
}

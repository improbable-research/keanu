package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Collections;
import java.util.Map;

public interface Differentiable {

    default PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        if (((Vertex)this).isObserved()) {
            return PartialDerivatives.OF_CONSTANT;
        } else {
            return PartialDerivatives.withRespectToSelf(((Vertex)this).getId(), ((Vertex)this).getShape());
        }
    }

    default Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        return Collections.singletonMap(
            ((Vertex)this),
            PartialDerivatives.withRespectToSelf(((Vertex)this).getId(), ((Vertex)this).getShape())
        );
    }

    default PartialDerivatives getDerivativeWrtLatents() {
        return Differentiator.forwardModeAutoDiff((Vertex & Differentiable) this);
    }
}

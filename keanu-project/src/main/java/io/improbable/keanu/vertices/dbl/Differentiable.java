package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Collections;
import java.util.Map;

public interface Differentiable {

    default PartialDerivative forwardModeAutoDifferentiation(Map<IVertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        if (((IVertex) this).isObserved()) {
            return PartialDerivative.EMPTY;
        } else {
            return withRespectToSelf(((IVertex) this).getShape());
        }
    }

    static PartialDerivative withRespectToSelf(long[] shape) {
        return new PartialDerivative(
            DoubleTensor.eye(TensorShape.getLength(shape)).reshape(TensorShape.concat(shape, shape))
        );
    }

    default Map<IVertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        return Collections.emptyMap();
    }
}

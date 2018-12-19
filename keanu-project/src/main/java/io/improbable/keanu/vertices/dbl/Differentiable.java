package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Collections;
import java.util.Map;

public interface Differentiable {

    default PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        if (((Vertex) this).isObserved()) {
            return PartialDerivative.EMPTY;
        } else {
            return withRespectToSelf(((Vertex) this).getShape());
        }
    }

    static PartialDerivative withRespectToSelf(long[] shape) {
        return new PartialDerivative(
            DoubleTensor.eye((int) TensorShape.getLength(shape)).reshape(TensorShape.concat(shape, shape))
        );
    }

    default Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        return Collections.emptyMap();
    }
}

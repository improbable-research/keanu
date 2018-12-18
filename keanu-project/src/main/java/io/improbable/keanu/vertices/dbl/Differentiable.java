package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Collections;
import java.util.Map;

public interface Differentiable {

    default PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInputs) {
        if (((Vertex) this).isObserved()) {
            return PartialDerivative.EMPTY;
        } else {
            return withRespectToSelf(((Vertex) this).getId(), ((Vertex) this).getShape());
        }
    }

    static PartialDerivative withRespectToSelf(VertexId withRespectTo, long[] shape) {
        return new PartialDerivative(
            withRespectTo,
            DoubleTensor.eye((int) TensorShape.getLength(shape)).reshape(TensorShape.concat(shape, shape))
        );
    }

    default Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        return Collections.emptyMap();
    }
}

package io.improbable.keanu.vertices.tensor.number.floating.dbl;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.Collections;
import java.util.Map;

public interface Differentiable {

    default ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        if (((Vertex) this).isObserved()) {
            return ForwardModePartialDerivative.EMPTY;
        } else {
            return wrtSelfOfSelf(((Vertex) this).getShape());
        }
    }

    static ReverseModePartialDerivative ofSelfWrtSelf(long[] shape) {
        return new ReverseModePartialDerivative(
            shape,
            DoubleTensor.eye(TensorShape.getLength(shape)).reshape(TensorShape.concat(shape, shape))
        );
    }

    static ForwardModePartialDerivative wrtSelfOfSelf(long[] shape) {
        return new ForwardModePartialDerivative(
            shape,
            DoubleTensor.eye(TensorShape.getLength(shape)).reshape(TensorShape.concat(shape, shape))
        );
    }

    default Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        return Collections.emptyMap();
    }
}

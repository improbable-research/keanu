package io.improbable.keanu.vertices.tensor.number.floating.dbl;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.VertexWrapper;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Map;

public class DoubleVertexWrapper extends VertexWrapper<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable {

    public static DoubleVertex wrapIfNeeded(Vertex<DoubleTensor, ?> vertex) {
        if (vertex instanceof DoubleVertex) {
            return (DoubleVertex) vertex;
        }

        if (vertex instanceof NonProbabilisticVertex) {
            return new DoubleVertexWrapper((NonProbabilisticVertex<DoubleTensor, ?>) vertex);
        } else {
            throw new IllegalStateException("Cannot wrap " + vertex.getClass().getCanonicalName() + " as DoubleVertex");
        }
    }

    public DoubleVertexWrapper(NonProbabilisticVertex<DoubleTensor, ?> vertex) {
        super(vertex);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        return ((Differentiable) unwrap()).reverseModeAutoDifferentiation(derivativeOfOutputWithRespectToSelf);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        return ((Differentiable) unwrap()).forwardModeAutoDifferentiation(derivativeOfParentsWithRespectToInput);
    }

}

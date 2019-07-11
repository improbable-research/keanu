package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import io.improbable.keanu.vertices.tensor.VertexWrapper;

import java.util.Map;

public class DoubleVertexWrapper extends VertexWrapper<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable {

    public DoubleVertexWrapper(NonProbabilisticVertex<DoubleTensor, DoubleVertex> vertex) {
        super(vertex);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        return ((Differentiable) getWrappedVertex()).reverseModeAutoDifferentiation(derivativeOfOutputWithRespectToSelf);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        return ((Differentiable) getWrappedVertex()).forwardModeAutoDifferentiation(derivativeOfParentsWithRespectToInput);
    }

}

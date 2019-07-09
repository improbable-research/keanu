package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;

@AllArgsConstructor
public class PartialsOf {

    @Getter
    private final IVertex<?> of;

    private final Map<VertexId, PartialDerivative> partials;

    public DoubleTensor withRespectTo(IVertex vertex) {
        return withRespectTo(vertex.getId());
    }

    public DoubleTensor withRespectTo(VertexId id) {
        return partials.get(id).get();
    }

    public Map<VertexId, PartialDerivative> asMap() {
        return partials;
    }

}

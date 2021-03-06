package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;

@AllArgsConstructor
public class PartialsOf {

    @Getter
    private final Vertex<?, ?> of;

    private final Map<VertexId, ReverseModePartialDerivative> partials;

    public DoubleTensor withRespectTo(Vertex vertex) {
        return withRespectTo(vertex.getId());
    }

    public DoubleTensor withRespectTo(VertexId id) {
        return partials.get(id).get();
    }

    public Map<VertexId, ReverseModePartialDerivative> asMap() {
        return partials;
    }

}

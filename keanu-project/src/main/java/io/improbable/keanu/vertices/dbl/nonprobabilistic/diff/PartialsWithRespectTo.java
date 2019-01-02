package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;

@AllArgsConstructor
public class PartialsWithRespectTo {

    @Getter
    private final Vertex wrt;

    private final Map<VertexId, PartialDerivative> partials;

    public PartialDerivative of(Vertex vertex) {
        return of(vertex.getId());
    }

    public PartialDerivative of(VertexId id) {
        return partials.get(id);
    }
}

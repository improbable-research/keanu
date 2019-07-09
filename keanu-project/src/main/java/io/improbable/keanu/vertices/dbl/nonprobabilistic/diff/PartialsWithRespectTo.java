package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;

@AllArgsConstructor
public class PartialsWithRespectTo {

    @Getter
    private final IVertex wrt;

    private final Map<VertexId, PartialDerivative> partials;

    public DoubleTensor of(IVertex vertex) {
        return of(vertex.getId());
    }

    public DoubleTensor of(VertexId id) {
        return partials.get(id).get();
    }
}

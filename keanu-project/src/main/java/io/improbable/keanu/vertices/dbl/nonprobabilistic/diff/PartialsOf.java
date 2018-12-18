package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;
import java.util.stream.Collectors;

@AllArgsConstructor
public class PartialsOf {

    @Getter
    private final Vertex<?> of;

    private final Map<VertexId, PartialDerivative> partials;

    public PartialDerivative withRespectTo(Vertex vertex) {
        return withRespectTo(vertex.getId());
    }

    public PartialDerivative withRespectTo(VertexId id) {
        return partials.get(id);
    }

    public Map<VertexId, DoubleTensor> asMap() {
        return partials.entrySet().stream()
            .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().getPartial()));
    }

}

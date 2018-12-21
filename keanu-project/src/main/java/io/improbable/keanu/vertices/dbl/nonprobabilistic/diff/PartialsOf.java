package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.HashMap;
import java.util.Map;

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
        final Map<VertexId, DoubleTensor> tensorMap = new HashMap<>();

        for (Map.Entry<VertexId, PartialDerivative> entry : partials.entrySet()) {
            tensorMap.put(entry.getKey(), entry.getValue().get());
        }

        return tensorMap;
    }

}

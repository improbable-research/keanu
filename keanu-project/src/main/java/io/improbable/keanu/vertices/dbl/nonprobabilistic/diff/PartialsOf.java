package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

@AllArgsConstructor
public class PartialsOf {

    @Getter
    private final Vertex<?> of;

    private final Map<VertexId, PartialDerivatives> partials;

    public PartialDerivatives withRespectTo(Vertex vertex) {
        return withRespectTo(vertex.getId());
    }

    public PartialDerivatives withRespectTo(VertexId id) {
        return partials.get(id);
    }

    public Map<VertexId, DoubleTensor> asMap() {
        return partials.entrySet().stream()
            .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().getValue()));
    }

    public void putWithRespectTo(VertexId id, PartialDerivatives partial) {
        partials.put(id, partial);
    }

    public PartialsOf add(PartialsOf other, Vertex<?> resultOf) {
        Map<VertexId, PartialDerivatives> clonedPartials = clonePartials(partials);

        for (Map.Entry<VertexId, PartialDerivatives> entry : other.partials.entrySet()) {
            VertexId id = entry.getKey();
            if (clonedPartials.containsKey(id)) {
                clonedPartials.put(id, clonedPartials.get(entry.getKey()).add(entry.getValue()));
            } else {
                clonedPartials.put(id, entry.getValue());
            }
        }

        return new PartialsOf(resultOf, clonedPartials);
    }

    private Map<VertexId, PartialDerivatives> clonePartials(Map<VertexId, PartialDerivatives> infinitesimals) {
        Map<VertexId, PartialDerivatives> clone = new HashMap<>();
        for (Map.Entry<VertexId, PartialDerivatives> entry : infinitesimals.entrySet()) {
            clone.put(entry.getKey(), entry.getValue());
        }
        return clone;
    }

}

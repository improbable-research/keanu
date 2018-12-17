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

    public void putWithRespectTo(VertexId id, PartialDerivative partial) {
        partials.put(id, partial);
    }

    public PartialsOf add(PartialsOf other, Vertex<?> resultOf) {
        Map<VertexId, PartialDerivative> clonedPartials = clonePartials(partials);

        for (Map.Entry<VertexId, PartialDerivative> entry : other.partials.entrySet()) {
            VertexId id = entry.getKey();
            if (clonedPartials.containsKey(id)) {

                DoubleTensor summation = clonedPartials.get(entry.getKey()).getPartial().plus(entry.getValue().getPartial());

                clonedPartials.put(id, new PartialDerivative(of != null ? of.getId() : null, summation));
            } else {
                clonedPartials.put(id, entry.getValue());
            }
        }

        return new PartialsOf(resultOf, clonedPartials);
    }

    private Map<VertexId, PartialDerivative> clonePartials(Map<VertexId, PartialDerivative> infinitesimals) {
        Map<VertexId, PartialDerivative> clone = new HashMap<>();
        for (Map.Entry<VertexId, PartialDerivative> entry : infinitesimals.entrySet()) {
            clone.put(entry.getKey(), entry.getValue());
        }
        return clone;
    }

}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.vertices.Vertex;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;

@AllArgsConstructor
public class PartialsWithRespectTo {

    @Getter
    private final Vertex wrt;

    private final Map<Vertex, PartialDerivative> partials;

    public PartialDerivative of(Vertex vertex) {
        return partials.get(vertex);
    }
}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.vertices.Vertex;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;

@AllArgsConstructor
public class PartialsOf {

    @Getter
    private final Vertex of;

    private final Map<Vertex, PartialDerivatives> partials;

    public PartialDerivatives withRespectTo(Vertex vertex) {
        return partials.get(vertex);
    }
}

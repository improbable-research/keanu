package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;

public class AndMultipleVertex extends BoolReduceVertex {
    public AndMultipleVertex(Collection<Vertex<Boolean>> input) {
        super(input, AndMultipleVertex::and);
    }

    private static Boolean and(Boolean a, Boolean b) {
        return a && b;
    }
}

package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;

public class OrMultipleVertex extends BoolReduceVertex {

    public OrMultipleVertex(Collection<Vertex<Boolean>> input) {
        super(input, OrMultipleVertex::or);
    }

    private static Boolean or(Boolean a, Boolean b) {
        return a | b;
    }
}

package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.Vertex;

public class OrBinaryVertex extends BoolBinaryOpVertex<Boolean, Boolean> {

    public OrBinaryVertex(Vertex<Boolean> a, Vertex<Boolean> b) {
        super(a, b);
    }

    @Override
    protected Boolean op(Boolean a, Boolean b) {
        return a || b;
    }
}

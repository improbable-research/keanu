package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.Vertex;

public class NotVertex extends BoolUnaryOpVertex<Boolean> {

    public NotVertex(Vertex<Boolean> a) {
        super(a);
    }

    @Override
    protected Boolean op(Boolean aBoolean) {
        return !aBoolean;
    }
}



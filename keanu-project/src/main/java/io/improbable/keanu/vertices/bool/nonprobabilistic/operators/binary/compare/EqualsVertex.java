package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class EqualsVertex<T> extends BoolBinaryOpVertex<T, T> {

    public EqualsVertex(Vertex<T> a, Vertex<T> b) {
        super(a, b);
    }

    @Override
    protected Boolean op(T a, T b) {
        return a.equals(b);
    }
}

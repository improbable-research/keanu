package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class NotEqualsVertex<T> extends BoolBinaryOpVertex<T, T> {

    public NotEqualsVertex(Vertex<T> a, Vertex<T> b) {
        super(a, b);
    }

    @Override
    public Boolean op(T a, T b) {
        return !a.equals(b);
    }

}

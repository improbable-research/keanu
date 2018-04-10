package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class LessThanVertex<A extends Number, B extends Number> extends BoolBinaryOpVertex<A, B> {

    public LessThanVertex(Vertex<A> a, Vertex<B> b) {
        super(a, b);
    }

    /**
     * Returns true if a is less than b
     */
    @Override
    public Boolean op(Number a, Number b) {
        return a.doubleValue() < b.doubleValue();
    }

}

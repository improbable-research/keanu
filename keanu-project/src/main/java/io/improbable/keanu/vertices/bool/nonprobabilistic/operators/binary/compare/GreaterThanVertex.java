package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class GreaterThanVertex<A extends Number, B extends Number> extends BoolBinaryOpVertex<A, B> {

    public GreaterThanVertex(Vertex<A> a, Vertex<B> b) {
        super(a, b);
    }

    /**
     * Returns true if a is greater than b
     */
    @Override
    public Boolean op(Number a, Number b) {
        return a.doubleValue() > b.doubleValue();
    }
}

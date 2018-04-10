package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class GreaterThanOrEqualVertex<A extends Number, B extends Number> extends BoolBinaryOpVertex<A, B> {

    public GreaterThanOrEqualVertex(Vertex<A> a, Vertex<B> b) {
        super(a, b);
    }

    /**
     * Returns true if a is greater than or equal to b
     */
    @Override
    public Boolean op(Number a, Number b) {
        return a.doubleValue() >= b.doubleValue();
    }

}

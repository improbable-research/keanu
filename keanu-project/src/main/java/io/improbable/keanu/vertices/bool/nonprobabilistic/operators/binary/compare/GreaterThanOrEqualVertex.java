package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class GreaterThanOrEqualVertex<A extends NumberTensor, B extends NumberTensor> extends BoolBinaryOpVertex<A, B> {

    public GreaterThanOrEqualVertex(Vertex<A> a, Vertex<B> b) {
        super(a, b, (l, r) -> l.toDouble().greaterThanOrEqual(r.toDouble()));
    }
}

package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class NotEqualsVertex<A extends Tensor, B extends Tensor> extends BoolBinaryOpVertex<A, B> {

    public NotEqualsVertex(Vertex<A> a, Vertex<B> b) {
        super(a, b, (l, r) -> l.elementwiseEquals(r).not());
    }
}

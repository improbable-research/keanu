package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class EqualsVertex<TENSOR extends Tensor> extends BoolBinaryOpVertex<TENSOR, TENSOR> {

    public EqualsVertex(Vertex<TENSOR> a, Vertex<TENSOR> b) {
        super(a, b, (l, r) -> l.elementwiseEquals(r));
    }
}

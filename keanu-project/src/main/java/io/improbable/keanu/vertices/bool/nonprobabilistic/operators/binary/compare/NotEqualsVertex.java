package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class NotEqualsVertex<A extends Tensor, B extends Tensor> extends BoolBinaryOpVertex<A, B> {

    public NotEqualsVertex(Vertex<A> a, Vertex<B> b) {
        super(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b,
            (l, r) -> l.elementwiseEquals(r).not());
    }
}

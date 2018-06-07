package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public class NotEqualsVertex<A extends Tensor, B extends Tensor> extends BoolBinaryOpVertex<A, B> {

    public NotEqualsVertex(Vertex<A> a, Vertex<B> b) {
        super(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b);
    }

    @Override
    public BooleanTensor op(A a, B b) {
        return a.elementwiseEquals(b).not();
    }

}

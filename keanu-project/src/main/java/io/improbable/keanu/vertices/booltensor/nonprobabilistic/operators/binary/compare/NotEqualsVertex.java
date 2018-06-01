package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class NotEqualsVertex<A extends Tensor, B extends Tensor> extends BoolBinaryOpVertex<A, B> {

    public NotEqualsVertex(Vertex<A> a, Vertex<B> b) {
        super(a, b);
    }

    @Override
    public BooleanTensor op(A a, B b) {
        return a.elementwiseEquals(b).not();
    }

}

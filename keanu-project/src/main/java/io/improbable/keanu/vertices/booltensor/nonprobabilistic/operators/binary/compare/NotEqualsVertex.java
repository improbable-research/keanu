package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class NotEqualsVertex<T, TA extends Tensor<T>, TB extends Tensor<T>> extends BoolBinaryOpVertex<TA, TB> {

    public NotEqualsVertex(Vertex<TA> a, Vertex<TB> b) {
        super(a, b);
    }

    @Override
    public BooleanTensor op(TA a, TB b) {
        return a.elementwiseEquals(b).not();
    }

}

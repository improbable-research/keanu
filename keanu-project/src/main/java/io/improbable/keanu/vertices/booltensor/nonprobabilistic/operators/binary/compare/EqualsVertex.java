package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.BoolBinaryOpVertex;

public class EqualsVertex<T, TENSOR extends Tensor<T>> extends BoolBinaryOpVertex<TENSOR, TENSOR> {

    public EqualsVertex(Vertex<TENSOR> a, Vertex<TENSOR> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(TENSOR a, TENSOR b) {
        return a.elementwiseEquals(b);
    }
}

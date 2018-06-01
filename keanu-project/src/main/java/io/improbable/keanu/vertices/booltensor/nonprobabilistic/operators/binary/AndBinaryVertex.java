package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary;


import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;

public class AndBinaryVertex extends BoolBinaryOpVertex<BooleanTensor, BooleanTensor> {

    public AndBinaryVertex(Vertex<BooleanTensor> a, Vertex<BooleanTensor> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(BooleanTensor a, BooleanTensor b) {
        return a.and(b);
    }
}

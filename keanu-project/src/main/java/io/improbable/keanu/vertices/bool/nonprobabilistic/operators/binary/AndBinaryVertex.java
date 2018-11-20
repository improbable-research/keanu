package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;


import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SaveableVertex;
import io.improbable.keanu.vertices.Vertex;

public class AndBinaryVertex extends BoolBinaryOpVertex<BooleanTensor, BooleanTensor> implements SaveableVertex {

    public AndBinaryVertex(@LoadParentVertex(A_NAME) Vertex<BooleanTensor> a,
                           @LoadParentVertex(B_NAME) Vertex<BooleanTensor> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(BooleanTensor l, BooleanTensor r) {
        return l.and(r);
    }
}

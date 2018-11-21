package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;


import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.Vertex;

@DisplayInformationForOutput(displayName = "AND")
public class AndBinaryVertex extends BoolBinaryOpVertex<BooleanTensor, BooleanTensor> {

    public AndBinaryVertex(@LoadParentVertex(A_NAME) Vertex<BooleanTensor> a,
                           @LoadParentVertex(B_NAME) Vertex<BooleanTensor> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(BooleanTensor l, BooleanTensor r) {
        return l.and(r);
    }
}

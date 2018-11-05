package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;


import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.util.dot.WriteDot;
import io.improbable.keanu.vertices.Vertex;

@WriteDot.Display(displayName = "AND")
public class AndBinaryVertex extends BoolBinaryOpVertex<BooleanTensor, BooleanTensor> {

    public AndBinaryVertex(Vertex<BooleanTensor> a, Vertex<BooleanTensor> b) {
        super(a, b);
    }

    @Override
    protected BooleanTensor op(BooleanTensor l, BooleanTensor r) {
        return l.and(r);
    }
}

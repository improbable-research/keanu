package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;

public class NotVertex extends BoolUnaryOpVertex<BooleanTensor> {

    public NotVertex(Vertex<BooleanTensor> a) {
        super(a.getShape(), a);
    }

    @Override
    protected BooleanTensor op(BooleanTensor aBoolean) {
        return aBoolean.not();
    }
}

package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.unary.BoolUnaryOpVertex;

public class NotVertex extends BoolUnaryOpVertex<BooleanTensor> {

    public NotVertex(Vertex<BooleanTensor> a) {
        super(a);
    }

    @Override
    protected BooleanTensor op(BooleanTensor aBoolean) {
        return aBoolean.not();
    }
}



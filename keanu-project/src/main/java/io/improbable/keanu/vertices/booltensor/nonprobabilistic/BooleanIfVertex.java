package io.improbable.keanu.vertices.booltensor.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.generictensor.nonprobabilistic.IfVertex;

public class BooleanIfVertex extends IfVertex<Boolean, BooleanTensor> {

    public BooleanIfVertex(int[] shape,
                           Vertex<? extends BooleanTensor> predicate,
                           Vertex<? extends BooleanTensor> thn,
                           Vertex<? extends BooleanTensor> els) {
        super(BooleanTensor.placeHolder(shape), predicate, thn, els);
    }

    @Override
    protected BooleanTensor op(BooleanTensor predicate, BooleanTensor thn, BooleanTensor els) {
        return predicate.setBooleanIf(thn, els);
    }
}

package io.improbable.keanu.vertices.dbltensor.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.generictensor.nonprobabilistic.IfVertex;

public class DoubleIfVertex extends IfVertex<Double, DoubleTensor> {

    public DoubleIfVertex(int[] shape,
                          Vertex<? extends BooleanTensor> predicate,
                          Vertex<? extends DoubleTensor> thn,
                          Vertex<? extends DoubleTensor> els) {
        super(DoubleTensor.placeHolder(shape), predicate, thn, els);
    }

    @Override
    protected DoubleTensor op(BooleanTensor predicate, DoubleTensor thn, DoubleTensor els) {
        return predicate.setDoubleIf(thn, els);
    }
}

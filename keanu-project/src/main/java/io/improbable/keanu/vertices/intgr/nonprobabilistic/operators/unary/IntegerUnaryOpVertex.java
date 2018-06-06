package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.NonProbabilisticInteger;

public abstract class IntegerUnaryOpVertex extends NonProbabilisticInteger {

    protected final IntegerVertex inputVertex;

    public IntegerUnaryOpVertex(int[] shape, IntegerVertex inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
        setValue(IntegerTensor.placeHolder(shape));
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    public IntegerTensor getDerivedValue() {
        return op(inputVertex.getValue());
    }

    protected abstract IntegerTensor op(IntegerTensor a);

}

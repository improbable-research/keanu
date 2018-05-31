package io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.intgrtensor.IntegerVertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.NonProbabilisticInteger;

public abstract class IntegerUnaryOpVertex extends NonProbabilisticInteger {

    protected final IntegerVertex inputVertex;

    public IntegerUnaryOpVertex(IntegerVertex inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
        setValue(IntegerTensor.placeHolder(inputVertex.getShape()));
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

package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.NonProbabilisticDoubleTensor;

public abstract class TensorDoubleUnaryOpVertex extends NonProbabilisticDoubleTensor {

    protected final DoubleTensorVertex inputVertex;

    public TensorDoubleUnaryOpVertex(DoubleTensorVertex inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
        setValue(DoubleTensor.placeHolder(inputVertex.getShape()));
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return op(inputVertex.getValue());
    }

    protected abstract DoubleTensor op(DoubleTensor a);

}

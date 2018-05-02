package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary;


import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.NonProbabilisticDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.DualNumber;

public abstract class BinaryOpVertex extends NonProbabilisticDoubleTensor {

    protected final DoubleTensorVertex a;
    protected final DoubleTensorVertex b;

    public BinaryOpVertex(DoubleTensorVertex a, DoubleTensorVertex b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public DoubleTensor sample() {
        return op(a.sample(), b.sample());
    }

    @Override
    public DoubleTensor lazyEval() {
        setValue(op(a.lazyEval(), b.lazyEval()));
        return getValue();
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract DualNumber getDualNumber();

    protected abstract DoubleTensor op(DoubleTensor a, DoubleTensor b);

}

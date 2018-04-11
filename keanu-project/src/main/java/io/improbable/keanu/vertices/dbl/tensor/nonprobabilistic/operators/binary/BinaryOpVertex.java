package io.improbable.keanu.vertices.dbl.tensor.nonprobabilistic.operators.binary;


import io.improbable.keanu.vertices.dbl.tensor.DoubleTensor;
import io.improbable.keanu.vertices.dbl.tensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbl.tensor.nonprobabilistic.NonProbabilisticDoubleTensor;
import io.improbable.keanu.vertices.dbl.tensor.nonprobabilistic.diff.DualNumber;

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

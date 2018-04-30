package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;

public abstract class DoubleUnaryOpVertex extends NonProbabilisticDouble {

    protected final DoubleVertex inputVertex;

    public DoubleUnaryOpVertex(DoubleVertex inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public Double sample() {
        return op(inputVertex.sample());
    }

    @Override
    public Double getDerivedValue() {
        return op(inputVertex.getValue());
    }

    protected abstract Double op(Double a);

}

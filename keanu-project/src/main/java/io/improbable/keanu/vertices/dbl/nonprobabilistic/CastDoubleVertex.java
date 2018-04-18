package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class CastDoubleVertex extends NonProbabilisticDouble {

    private final Vertex<Double> inputVertex;

    public CastDoubleVertex(Vertex<Double> inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public Double sample() {
        return inputVertex.sample();
    }

    @Override
    public Double lazyEval() {
        setValue(inputVertex.lazyEval());
        return getValue();
    }

    @Override
    public Double getDerivedValue() {
        return inputVertex.getValue();
    }

    @Override
    public DualNumber getDualNumber() {
        throw new UnsupportedOperationException("CastDoubleVertex is non-differentiable");
    }
}

package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Map;

public class CastDoubleVertex extends NonProbabilisticDouble {

    private final Vertex<? extends Number> inputVertex;

    public CastDoubleVertex(Vertex<? extends Number> inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public Double sample(KeanuRandom random) {
        return inputVertex.sample(random).doubleValue();
    }

    @Override
    public Double getDerivedValue() {
        return inputVertex.getValue().doubleValue();
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        throw new UnsupportedOperationException("CastDoubleVertex is non-differentiable");
    }
}

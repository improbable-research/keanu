package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class CastDoubleVertex extends NonProbabilisticDouble {

    private final Vertex<? extends NumberTensor> inputVertex;

    public CastDoubleVertex(Vertex<? extends NumberTensor> inputVertex) {
        super(v -> ((CastDoubleVertex)v).inputVertex.getValue().toDouble());
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return inputVertex.sample(random).toDouble();
    }
}

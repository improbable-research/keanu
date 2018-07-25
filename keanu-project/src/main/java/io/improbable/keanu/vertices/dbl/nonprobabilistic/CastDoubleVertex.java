package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class CastDoubleVertex extends DoubleVertex {

    private final Vertex<? extends NumberTensor> inputVertex;

    public CastDoubleVertex(Vertex<? extends NumberTensor> inputVertex) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((CastDoubleVertex)v).inputVertex.getValue().toDouble()),
            Observable.observableTypeFor(CastDoubleVertex.class)
        );
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return inputVertex.sample(random).toDouble();
    }
}

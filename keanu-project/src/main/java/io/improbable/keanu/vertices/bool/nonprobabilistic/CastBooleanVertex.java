package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class CastBooleanVertex extends BooleanVertex {

    private final Vertex<? extends BooleanTensor> inputVertex;

    public CastBooleanVertex(Vertex<? extends BooleanTensor> inputVertex) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((CastBooleanVertex) v).inputVertex.getValue()),
            Observable.observableTypeFor(CastBooleanVertex.class)
        );
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return inputVertex.sample(random);
    }

}


package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observation;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class CastBoolVertex extends BoolVertex {

    private final Vertex<? extends BooleanTensor> inputVertex;

    public CastBoolVertex(Vertex<? extends BooleanTensor> inputVertex) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((CastBoolVertex) v).inputVertex.getValue()),
            new Observation<>()
        );
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return inputVertex.sample(random);
    }

}


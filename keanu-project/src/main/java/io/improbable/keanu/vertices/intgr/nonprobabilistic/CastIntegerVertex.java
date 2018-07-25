package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class CastIntegerVertex extends IntegerVertex {

    private final Vertex<IntegerTensor> inputVertex;

    public CastIntegerVertex(Vertex<IntegerTensor> inputVertex) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((CastIntegerVertex) v).inputVertex.getValue()),
            Observable.observableTypeFor(CastIntegerVertex.class)
        );
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return inputVertex.sample(random);
    }

}

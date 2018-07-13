package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class BooleanIfVertex extends BoolVertex {

    private final Vertex<? extends BooleanTensor> predicate;
    private final Vertex<? extends BooleanTensor> thn;
    private final Vertex<? extends BooleanTensor> els;

    public BooleanIfVertex(int[] shape,
                           Vertex<? extends BooleanTensor> predicate,
                           Vertex<? extends BooleanTensor> thn,
                           Vertex<? extends BooleanTensor> els) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((BooleanIfVertex) v).op(predicate.getValue(), thn.getValue(), els.getValue())),
            Observable.observableTypeFor(BooleanIfVertex.class)
        );
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
        setValue(BooleanTensor.placeHolder(shape));
    }

    protected BooleanTensor op(BooleanTensor predicate, BooleanTensor thn, BooleanTensor els) {
        return predicate.setBooleanIf(thn, els);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(predicate.sample(random), thn.sample(random), els.sample(random));
    }
}

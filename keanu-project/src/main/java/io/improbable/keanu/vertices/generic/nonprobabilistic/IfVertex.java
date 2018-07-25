package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.GenericVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class IfVertex<T> extends GenericVertex<Tensor<T>> {

    private final Vertex<? extends BooleanTensor> predicate;
    private final Vertex<? extends Tensor<T>> thn;
    private final Vertex<? extends Tensor<T>> els;

    public IfVertex(int[] shape,
                    Vertex<? extends BooleanTensor> predicate,
                    Vertex<? extends Tensor<T>> thn,
                    Vertex<? extends Tensor<T>> els) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((IfVertex) v).op(predicate.getValue(), thn.getValue(), els.getValue())),
            Observable.observableTypeFor(IfVertex.class)
        );
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
        setValue(Tensor.placeHolder(shape));
    }

    @Override
    public Tensor<T> sample(KeanuRandom random) {
        return op(predicate.sample(random), thn.sample(random), els.sample(random));
    }

    private Tensor<T> op(BooleanTensor predicate, Tensor<T> thn, Tensor<T> els) {
        return predicate.setIf(thn, els);
    }
}

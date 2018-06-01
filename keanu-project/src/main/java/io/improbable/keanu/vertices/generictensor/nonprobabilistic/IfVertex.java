package io.improbable.keanu.vertices.generictensor.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class IfVertex<T> extends NonProbabilistic<Tensor<T>> {

    private final Vertex<? extends BooleanTensor> predicate;
    private final Vertex<? extends Tensor<T>> thn;
    private final Vertex<? extends Tensor<T>> els;

    public IfVertex(Vertex<? extends BooleanTensor> predicate,
                    Vertex<? extends Tensor<T>> thn,
                    Vertex<? extends Tensor<T>> els) {

        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    @Override
    public Tensor<T> sample(KeanuRandom random) {
        return op(predicate.sample(random), thn.sample(random), els.sample(random));
    }

    @Override
    public Tensor<T> getDerivedValue() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    protected Tensor<T> op(BooleanTensor predicate, Tensor<T> thn, Tensor<T> els) {
        return predicate.setIf(thn, els);
    }
}

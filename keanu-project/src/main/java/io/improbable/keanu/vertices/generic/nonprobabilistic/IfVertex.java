package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class IfVertex<T> extends Vertex<Tensor<T>> implements NonProbabilistic<Tensor<T>> {

    private final Vertex<? extends BooleanTensor> predicate;
    private final Vertex<? extends Tensor<T>> thn;
    private final Vertex<? extends Tensor<T>> els;

    public IfVertex(long[] shape,
                    Vertex<? extends BooleanTensor> predicate,
                    Vertex<? extends Tensor<T>> thn,
                    Vertex<? extends Tensor<T>> els) {
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
        setValue(GenericTensor.placeHolder(shape));
    }

    private Tensor<T> op(BooleanTensor predicate, Tensor<T> thn, Tensor<T> els) {
        return predicate.where(thn, els);
    }

    @Override
    public Tensor<T> sample(KeanuRandom random) {
        return op(predicate.sample(random), thn.sample(random), els.sample(random));
    }

    @Override
    public Tensor<T> calculate() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }
}

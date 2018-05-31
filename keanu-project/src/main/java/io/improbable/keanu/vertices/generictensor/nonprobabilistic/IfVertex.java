package io.improbable.keanu.vertices.generictensor.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public abstract class IfVertex<T, TENSOR extends Tensor<T>> extends NonProbabilistic<T, TENSOR> {

    private final Vertex<? extends BooleanTensor> predicate;
    private final Vertex<? extends TENSOR> thn;
    private final Vertex<? extends TENSOR> els;

    public IfVertex(TENSOR placeHolder,
                    Vertex<? extends BooleanTensor> predicate,
                    Vertex<? extends TENSOR> thn,
                    Vertex<? extends TENSOR> els) {
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
        setValue(placeHolder);
    }

    @Override
    public TENSOR sample(KeanuRandom random) {
        return op(predicate.sample(random), thn.sample(random), els.sample(random));
    }

    @Override
    public TENSOR getDerivedValue() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    protected abstract TENSOR op(BooleanTensor predicate, TENSOR thn, TENSOR els);

}

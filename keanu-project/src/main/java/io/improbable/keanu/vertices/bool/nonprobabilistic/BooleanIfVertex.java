package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class BooleanIfVertex extends BoolVertex implements NonProbabilistic<BooleanTensor> {

    private final Vertex<? extends BooleanTensor> predicate;
    private final Vertex<? extends BooleanTensor> thn;
    private final Vertex<? extends BooleanTensor> els;

    public BooleanIfVertex(Vertex<? extends BooleanTensor> predicate,
                           Vertex<? extends BooleanTensor> thn,
                           Vertex<? extends BooleanTensor> els) {
        super(TensorShapeValidation.checkTernaryConditionShapeIsValid(thn.getShape(), els.getShape(), predicate.getShape()));
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    protected BooleanTensor op(BooleanTensor predicate, BooleanTensor thn, BooleanTensor els) {
        return predicate.booleanWhere(thn, els);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(predicate.sample(random), thn.sample(random), els.sample(random));
    }

    @Override
    public BooleanTensor calculate() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }
}

package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class IfVertex<T> extends Vertex<Tensor<T>> implements NonProbabilistic<Tensor<T>> {

    private final static String PREDICATE_NAME = "predicate";
    private final static String THEN_NAME = "then";
    private final static String ELSE_NAME = "else";

    private final Vertex<? extends BooleanTensor> predicate;
    private final Vertex<? extends Tensor<T>> thn;
    private final Vertex<? extends Tensor<T>> els;

    public IfVertex(@LoadVertexParam(PREDICATE_NAME) Vertex<? extends BooleanTensor> predicate,
                    @LoadVertexParam(THEN_NAME) Vertex<? extends Tensor<T>> thn,
                    @LoadVertexParam(ELSE_NAME) Vertex<? extends Tensor<T>> els) {
        super(TensorShapeValidation.checkTernaryConditionShapeIsValid(predicate.getShape(), thn.getShape(), els.getShape()));
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
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

    @SaveVertexParam(PREDICATE_NAME)
    public Vertex<? extends BooleanTensor> getPredicateParam() {
        return predicate;
    }

    @SaveVertexParam(THEN_NAME)
    public Vertex<? extends Tensor<T>> getThn() {
        return thn;
    }

    @SaveVertexParam(ELSE_NAME)
    public Vertex<? extends Tensor<T>> getEls() {
        return els;
    }
}

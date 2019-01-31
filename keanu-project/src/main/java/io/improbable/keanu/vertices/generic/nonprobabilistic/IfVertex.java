package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

public class IfVertex<T> extends GenericTensorVertex<Tensor<T>> implements NonProbabilistic<Tensor<T>> {

    private final static String PREDICATE_NAME = "predicate";
    private final static String THEN_NAME = "then";
    private final static String ELSE_NAME = "else";

    private final BooleanVertex predicate;
    private final Vertex<? extends Tensor<T>> thn;
    private final Vertex<? extends Tensor<T>> els;

    public IfVertex(@LoadVertexParam(PREDICATE_NAME) BooleanVertex predicate,
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
    public Tensor<T> calculate() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    @SaveVertexParam(PREDICATE_NAME)
    public BooleanVertex getPredicate() {
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

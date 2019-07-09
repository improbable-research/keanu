package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

public class IfVertex<T> extends GenericTensorVertex<T> implements NonProbabilistic<GenericTensor<T>> {

    private final static String PREDICATE_NAME = "predicate";
    private final static String THEN_NAME = "then";
    private final static String ELSE_NAME = "else";

    private final BooleanVertex predicate;
    private final IVertex<GenericTensor<T>> thn;
    private final IVertex<GenericTensor<T>> els;

    public IfVertex(@LoadVertexParam(PREDICATE_NAME) BooleanVertex predicate,
                    @LoadVertexParam(THEN_NAME) IVertex<GenericTensor<T>> thn,
                    @LoadVertexParam(ELSE_NAME) IVertex<GenericTensor<T>> els) {
        super(TensorShapeValidation.checkTernaryConditionShapeIsValid(predicate.getShape(), thn.getShape(), els.getShape()));
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    private GenericTensor<T> op(BooleanTensor predicate, GenericTensor<T> thn, GenericTensor<T> els) {
        return predicate.where(thn, els);
    }

    @Override
    public GenericTensor<T> calculate() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    @SaveVertexParam(PREDICATE_NAME)
    public BooleanVertex getPredicate() {
        return predicate;
    }

    @SaveVertexParam(THEN_NAME)
    public IVertex<GenericTensor<T>> getThn() {
        return thn;
    }

    @SaveVertexParam(ELSE_NAME)
    public IVertex<GenericTensor<T>> getEls() {
        return els;
    }
}

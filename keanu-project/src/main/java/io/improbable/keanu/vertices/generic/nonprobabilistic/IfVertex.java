package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

public class IfVertex<T, TENSOR extends Tensor<T, TENSOR>> extends GenericTensorVertex<T, TENSOR> implements NonProbabilistic<TENSOR> {

    private final static String PREDICATE_NAME = "predicate";
    private final static String THEN_NAME = "then";
    private final static String ELSE_NAME = "else";

    private final BooleanVertex predicate;
    private final Vertex<TENSOR> thn;
    private final Vertex<TENSOR> els;

    public IfVertex(@LoadVertexParam(PREDICATE_NAME) BooleanVertex predicate,
                    @LoadVertexParam(THEN_NAME) Vertex<TENSOR> thn,
                    @LoadVertexParam(ELSE_NAME) Vertex<TENSOR> els) {
        super(TensorShapeValidation.checkTernaryConditionShapeIsValid(predicate.getShape(), thn.getShape(), els.getShape()));
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    private TENSOR op(BooleanTensor predicate, TENSOR thn, TENSOR els) {
        return predicate.where(thn, els);
    }

    @Override
    public TENSOR calculate() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    @SaveVertexParam(PREDICATE_NAME)
    public BooleanVertex getPredicate() {
        return predicate;
    }

    @SaveVertexParam(THEN_NAME)
    public Vertex<? extends Tensor<T, TENSOR>> getThn() {
        return thn;
    }

    @SaveVertexParam(ELSE_NAME)
    public Vertex<? extends Tensor<T, TENSOR>> getEls() {
        return els;
    }
}

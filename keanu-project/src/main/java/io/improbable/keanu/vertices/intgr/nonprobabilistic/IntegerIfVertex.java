package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerIfVertex extends IntegerVertex implements NonProbabilistic<IntegerTensor> {

    protected static final String PREDICATE_NAME = "predicate";
    protected static final String THEN_NAME = "then";
    protected static final String ELSE_NAME = "else";
    private final Vertex<? extends BooleanTensor> predicate;
    private final IntegerVertex thn;
    private final IntegerVertex els;

    @ExportVertexToPythonBindings
    public IntegerIfVertex(@LoadVertexParam(PREDICATE_NAME) Vertex<? extends BooleanTensor> predicate,
                           @LoadVertexParam(THEN_NAME) IntegerVertex thn,
                           @LoadVertexParam(ELSE_NAME) IntegerVertex els) {
        super(TensorShapeValidation.checkTernaryConditionShapeIsValid(predicate.getShape(), thn.getShape(), els.getShape()));
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    @SaveVertexParam(PREDICATE_NAME)
    public Vertex<? extends BooleanTensor> getPredicate() {
        return predicate;
    }

    @SaveVertexParam(THEN_NAME)
    public IntegerVertex getThn() {
        return thn;
    }

    @SaveVertexParam(ELSE_NAME)
    public IntegerVertex getEls() {
        return els;
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op(predicate.sample(random), thn.sample(random), els.sample(random));
    }

    @Override
    public IntegerTensor calculate() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    private IntegerTensor op(BooleanTensor predicate, IntegerTensor thn, IntegerTensor els) {
        return predicate.integerWhere(thn, els);
    }

}

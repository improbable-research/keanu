package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public class BooleanIfVertex extends BooleanVertex implements NonProbabilistic<BooleanTensor> {

    private final BooleanVertex predicate;
    private final BooleanVertex thn;
    private final BooleanVertex els;
    private final static String PRED_NAME = "predicate";
    private final static String THN_NAME = "then";
    private final static String ELS_NAME = "else";

    @ExportVertexToPythonBindings
    public BooleanIfVertex(@LoadVertexParam(PRED_NAME) BooleanVertex predicate,
                           @LoadVertexParam(THN_NAME) BooleanVertex thn,
                           @LoadVertexParam(ELS_NAME) BooleanVertex els) {
        super(TensorShapeValidation.checkTernaryConditionShapeIsValid(predicate.getShape(), thn.getShape(), els.getShape()));
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    protected BooleanTensor op(BooleanTensor predicate, BooleanTensor thn, BooleanTensor els) {
        return predicate.booleanWhere(thn, els);
    }

    @Override
    public BooleanTensor calculate() {
        return op(predicate.getValue(), thn.getValue(), els.getValue());
    }

    @SaveVertexParam(PRED_NAME)
    public BooleanVertex getPredicate() {
        return predicate;
    }

    @SaveVertexParam(THN_NAME)
    public BooleanVertex getThn() {
        return thn;
    }

    @SaveVertexParam(ELS_NAME)
    public BooleanVertex getEls() {
        return els;
    }
}

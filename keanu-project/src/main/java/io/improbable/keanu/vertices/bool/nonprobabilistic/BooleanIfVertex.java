package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class BooleanIfVertex extends BoolVertex implements NonProbabilistic<BooleanTensor> {

    private final Vertex<? extends BooleanTensor> predicate;
    private final Vertex<? extends BooleanTensor> thn;
    private final Vertex<? extends BooleanTensor> els;
    private final static String PRED_NAME = "predicate";
    private final static String THN_NAME = "then";
    private final static String ELS_NAME = "else";

    public BooleanIfVertex(long[] shape,
                           Vertex<? extends BooleanTensor> predicate,
                           Vertex<? extends BooleanTensor> thn,
                           Vertex<? extends BooleanTensor> els) {
        super(shape);
        this.predicate = predicate;
        this.thn = thn;
        this.els = els;
        setParents(predicate, thn, els);
    }

    @ExportVertexToPythonBindings
    public BooleanIfVertex(@LoadVertexParam(PRED_NAME) Vertex<? extends BooleanTensor> predicate,
                           @LoadVertexParam(THN_NAME) Vertex<? extends BooleanTensor> thn,
                           @LoadVertexParam(ELS_NAME) Vertex<? extends BooleanTensor> els) {
        this(els.getShape(), predicate, thn, els);
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

    @SaveVertexParam(PRED_NAME)
    public Vertex<? extends BooleanTensor> getPredicate() {
        return predicate;
    }

    @SaveVertexParam(THN_NAME)
    public Vertex<? extends BooleanTensor> getThn() {
        return thn;
    }

    @SaveVertexParam(ELS_NAME)
    public Vertex<? extends BooleanTensor> getEls() {
        return els;
    }
}

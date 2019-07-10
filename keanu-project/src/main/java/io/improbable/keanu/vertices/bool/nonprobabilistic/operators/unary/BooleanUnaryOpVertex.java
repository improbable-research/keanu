package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public abstract class BooleanUnaryOpVertex extends VertexImpl<BooleanTensor, BooleanVertex> implements BooleanVertex, NonProbabilistic<BooleanTensor>, VertexUnaryOp<BooleanVertex> {

    protected final BooleanVertex inputVertex;
    protected final static String INPUT_NAME = "inputVertex";

    public BooleanUnaryOpVertex(BooleanVertex inputVertex) {
        this(inputVertex.getShape(), inputVertex);
    }

    public BooleanUnaryOpVertex(long[] shape, BooleanVertex inputVertex) {
        super(shape);
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @SaveVertexParam(INPUT_NAME)
    public BooleanVertex getInputVertex() {
        return inputVertex;
    }

    @Override
    public BooleanTensor calculate() {
        return op(inputVertex.getValue());
    }

    protected abstract BooleanTensor op(BooleanTensor value);

}
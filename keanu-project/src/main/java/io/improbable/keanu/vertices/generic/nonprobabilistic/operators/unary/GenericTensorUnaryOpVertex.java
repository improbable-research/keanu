package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

public abstract class GenericTensorUnaryOpVertex<IN, OUT> extends VertexImpl<GenericTensor<OUT>, GenericTensorVertex<OUT>> implements GenericTensorVertex<OUT>, NonProbabilistic<GenericTensor<OUT>> {

    protected static final String INPUT_NAME = "inputVertex";

    protected final Vertex<GenericTensor<IN>, ?> inputVertex;

    public GenericTensorUnaryOpVertex(long[] shape, Vertex<GenericTensor<IN>, ?> inputVertex) {
        super(shape);
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public GenericTensor<OUT> calculate() {
        return op(inputVertex.getValue());
    }

    protected abstract GenericTensor<OUT> op(GenericTensor<IN> a);

    @SaveVertexParam(INPUT_NAME)
    public Vertex<GenericTensor<IN>, ?> getInputVertex() {
        return inputVertex;
    }
}

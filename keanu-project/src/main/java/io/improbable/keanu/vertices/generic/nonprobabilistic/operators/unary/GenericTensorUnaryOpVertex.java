package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

public abstract class GenericTensorUnaryOpVertex<IN, OUT> extends GenericTensorVertex<OUT> implements NonProbabilistic<Tensor<OUT>> {

    protected static final String INPUT_NAME = "inputVertex";

    protected final Vertex<Tensor<IN>> inputVertex;

    public GenericTensorUnaryOpVertex(long[] shape, Vertex<Tensor<IN>> inputVertex) {
        super(shape);
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public Tensor<OUT> calculate() {
        return op(inputVertex.getValue());
    }

    protected abstract Tensor<OUT> op(Tensor<IN> a);

    @SaveVertexParam(INPUT_NAME)
    public Vertex<Tensor<IN>> getInputVertex() {
        return inputVertex;
    }
}

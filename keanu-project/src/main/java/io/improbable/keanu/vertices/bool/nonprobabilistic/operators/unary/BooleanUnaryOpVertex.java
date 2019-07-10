package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public abstract class BooleanUnaryOpVertex<T extends Tensor> extends VertexImpl<BooleanTensor> implements BooleanVertex, NonProbabilistic<BooleanTensor>, VertexUnaryOp<Vertex<T>> {

    protected final Vertex<T> inputVertex;
    protected final static String INPUT_NAME = "inputVertex";

    public BooleanUnaryOpVertex(Vertex<T> inputVertex) {
        this(inputVertex.getShape(), inputVertex);
    }

    public BooleanUnaryOpVertex(long[] shape, Vertex<T> inputVertex) {
        super(shape);
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @SaveVertexParam(INPUT_NAME)
    public Vertex<T> getInputVertex() {
        return inputVertex;
    }

    @Override
    public BooleanTensor calculate() {
        return op(inputVertex.getValue());
    }

    protected abstract BooleanTensor op(T value);

}
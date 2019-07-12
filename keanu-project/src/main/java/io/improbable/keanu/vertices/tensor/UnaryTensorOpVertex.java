package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexUnaryOp;

public abstract class UnaryTensorOpVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends VertexImpl<TENSOR, VERTEX> implements NonProbabilistic<TENSOR>, VertexUnaryOp<TensorVertex<T, TENSOR, VERTEX>> {

    protected final TensorVertex<T, TENSOR, VERTEX> inputVertex;
    private final Class<?> type;

    protected static final String INPUT_NAME = "inputVertex";
    protected static final String TYPE_NAME = "type";

    /**
     * A vertex that performs a user defined operation on a singe input vertex
     *
     * @param inputVertex the input vertex
     */
    public UnaryTensorOpVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex, Class<?> type) {
        this(inputVertex.getShape(), inputVertex, type);
    }

    /**
     * A vertex that performs a user defined operation on a singe input vertex
     *
     * @param shape       the shape of the tensor
     * @param inputVertex the input vertex
     */
    public UnaryTensorOpVertex(long[] shape, TensorVertex<T, TENSOR, VERTEX> inputVertex, Class<?> type) {
        super(shape);
        this.inputVertex = inputVertex;
        this.type = type;
        setParents(inputVertex);
    }

    @Override
    public TENSOR calculate() {
        return op(inputVertex.getValue());
    }

    protected abstract TENSOR op(TENSOR value);

    @SaveVertexParam(INPUT_NAME)
    public TensorVertex<T, TENSOR, VERTEX> getInputVertex() {
        return inputVertex;
    }

    public Class<?> ofType(){
        return type;
    }
}

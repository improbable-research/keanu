package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexUnaryOp;

public abstract class UnaryTensorOpVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends VertexImpl<TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, VertexUnaryOp<TensorVertex<T, TENSOR, VERTEX>>, TensorVertex<T, TENSOR, VERTEX> {

    protected final TensorVertex<T, TENSOR, VERTEX> inputVertex;
    private final Class<?> type;

    protected static final String INPUT_NAME = "inputVertex";

    public UnaryTensorOpVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        this(inputVertex.getShape(), inputVertex, inputVertex.ofType());
    }

    /**
     * A vertex that performs a user defined operation on a singe input vertex
     *
     * @param inputVertex the input vertex
     * @param type        type of tensor outputted
     */
    public UnaryTensorOpVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex, Class<?> type) {
        this(inputVertex.getShape(), inputVertex, type);
    }

    /**
     * A vertex that performs a user defined operation on a singe input vertex
     *
     * @param shape       the shape of the tensor
     * @param inputVertex the input vertex
     * @param type        type of tensor outputted
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

    public Class<?> ofType() {
        return type;
    }

    @Override
    public VERTEX wrap(NonProbabilisticVertex<TENSOR, VERTEX> vertex) {
        return null;
    }
}

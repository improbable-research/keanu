package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.BooleanVertex;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;

public abstract class BinaryTensorOpVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends VertexImpl<TENSOR, VERTEX>
    implements NonProbabilisticVertex<TENSOR, VERTEX>, VertexBinaryOp<TensorVertex<T, TENSOR, VERTEX>, TensorVertex<T, TENSOR, VERTEX>>, TensorVertex<T, TENSOR, VERTEX> {

    protected final TensorVertex<T, TENSOR, VERTEX> left;
    protected final TensorVertex<T, TENSOR, VERTEX> right;
    private final Class<?> type;

    protected static final String LEFT_NAME = "left";
    protected static final String RIGHT_NAME = "right";

    /**
     * A vertex that performs a user defined operation on two input vertices
     *
     * @param left  first input vertex
     * @param right second input vertex
     */
    public BinaryTensorOpVertex(TensorVertex<T, TENSOR, VERTEX> left, TensorVertex<T, TENSOR, VERTEX> right, Class<?> type) {
        this(getBroadcastResultShape(left.getShape(), right.getShape()), left, right, type);
    }

    /**
     * A vertex that performs a user defined operation on two input vertices
     *
     * @param shape the shape of the tensor
     * @param left  first input vertex
     * @param right second input vertex
     */
    public BinaryTensorOpVertex(long[] shape, TensorVertex<T, TENSOR, VERTEX> left, TensorVertex<T, TENSOR, VERTEX> right, Class<?> type) {
        super(shape);
        this.left = left;
        this.right = right;
        this.type = type;
        setParents(left, right);
    }

    @Override
    public TENSOR calculate() {
        return op(left.getValue(), right.getValue());
    }

    protected abstract TENSOR op(TENSOR l, TENSOR r);

    @Override
    @SaveVertexParam(LEFT_NAME)
    public TensorVertex<T, TENSOR, VERTEX> getLeft() {
        return left;
    }

    @Override
    @SaveVertexParam(RIGHT_NAME)
    public TensorVertex<T, TENSOR, VERTEX> getRight() {
        return right;
    }

    @Override
    public Class<?> ofType() {
        return type;
    }

    @Override
    public VERTEX wrap(NonProbabilisticVertex<TENSOR, VERTEX> vertex) {
        return null;
    }

}

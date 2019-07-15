package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;

public abstract class IntegerBinaryOpVertex<
    A, TENSORA extends Tensor<A, TENSORA>, VERTEXA extends TensorVertex<A, TENSORA, VERTEXA>,
    B, TENSORB extends Tensor<B, TENSORB>, VERTEXB extends TensorVertex<B, TENSORB, VERTEXB>
    > extends VertexImpl<IntegerTensor, IntegerVertex>
    implements IntegerVertex, NonProbabilistic<IntegerTensor>, VertexBinaryOp<TensorVertex<A, TENSORA, VERTEXA>, TensorVertex<B, TENSORB, VERTEXB>> {

    protected final TensorVertex<A, TENSORA, VERTEXA> left;
    protected final TensorVertex<B, TENSORB, VERTEXB> right;
    protected static final String LEFT_NAME = "left";
    protected static final String RIGHT_NAME = "right";

    /**
     * A vertex that performs a user defined operation on two input vertices
     *
     * @param left  first input vertex
     * @param right second input vertex
     */
    public IntegerBinaryOpVertex(TensorVertex<A, TENSORA, VERTEXA> left, TensorVertex<B, TENSORB, VERTEXB> right) {
        this(getBroadcastResultShape(left.getShape(), right.getShape()), left, right);
    }

    /**
     * A vertex that performs a user defined operation on two input vertices
     *
     * @param shape the shape of the tensor
     * @param left  first input vertex
     * @param right second input vertex
     */
    public IntegerBinaryOpVertex(long[] shape, TensorVertex<A, TENSORA, VERTEXA> left, TensorVertex<B, TENSORB, VERTEXB> right) {
        super(shape);
        this.left = left;
        this.right = right;
        setParents(left, right);
    }

    @Override
    public IntegerTensor calculate() {
        return op(left.getValue(), right.getValue());
    }

    protected abstract IntegerTensor op(TENSORA l, TENSORB r);

    @Override
    @SaveVertexParam(LEFT_NAME)
    public TensorVertex<A, TENSORA, VERTEXA> getLeft() {
        return left;
    }

    @Override
    @SaveVertexParam(RIGHT_NAME)
    public TensorVertex<B, TENSORB, VERTEXB> getRight() {
        return right;
    }
}

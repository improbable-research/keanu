package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;

public abstract class IntegerBinaryOpVertex extends VertexImpl<IntegerTensor> implements IntegerVertex, NonProbabilistic<IntegerTensor>, VertexBinaryOp<IntegerVertex, IntegerVertex> {

    protected final IntegerVertex left;
    protected final IntegerVertex right;
    protected static final String LEFT_NAME = "left";
    protected static final String RIGHT_NAME = "right";

    /**
     * A vertex that performs a user defined operation on two input vertices
     *
     * @param left  first input vertex
     * @param right second input vertex
     */
    public IntegerBinaryOpVertex(IntegerVertex left, IntegerVertex right) {
        this(getBroadcastResultShape(left.getShape(), right.getShape()), left, right);
    }

    /**
     * A vertex that performs a user defined operation on two input vertices
     *
     * @param shape the shape of the tensor
     * @param left  first input vertex
     * @param right second input vertex
     */
    public IntegerBinaryOpVertex(long[] shape, IntegerVertex left, IntegerVertex right) {
        super(shape);
        this.left = left;
        this.right = right;
        setParents(left, right);
    }

    @Override
    public IntegerTensor calculate() {
        return op(left.getValue(), right.getValue());
    }

    protected abstract IntegerTensor op(IntegerTensor l, IntegerTensor r);

    @Override
    @SaveVertexParam(LEFT_NAME)
    public IntegerVertex getLeft() {
        return left;
    }

    @Override
    @SaveVertexParam(RIGHT_NAME)
    public IntegerVertex getRight() {
        return right;
    }
}

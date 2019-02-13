package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;


import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;

public abstract class DoubleBinaryOpVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor>, VertexBinaryOp<DoubleVertex, DoubleVertex> {

    protected final DoubleVertex left;
    protected final DoubleVertex right;
    protected static final String LEFT_NAME = "left";
    protected static final String RIGHT_NAME = "right";

    /**
     * A vertex that performs a user defined operation on two vertices
     *
     * @param left  a vertex
     * @param right a vertex
     */
    public DoubleBinaryOpVertex(
        DoubleVertex left, DoubleVertex right) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()),
            left, right);
    }

    /**
     * A vertex that performs a user defined operation on two vertices
     *
     * @param shape the shape of the resulting vertex
     * @param left  a vertex
     * @param right a vertex
     */
    public DoubleBinaryOpVertex(long[] shape, DoubleVertex left, DoubleVertex right) {
        super(shape);
        this.left = left;
        this.right = right;
        setParents(left, right);
    }

    @Override
    public DoubleTensor calculate() {
        return op(left.getValue(), right.getValue());
    }

    @Override
    @SaveVertexParam(LEFT_NAME)
    public DoubleVertex getLeft() {
        return left;
    }

    @Override
    @SaveVertexParam(RIGHT_NAME)
    public DoubleVertex getRight() {
        return right;
    }

    protected abstract DoubleTensor op(DoubleTensor l, DoubleTensor r);
}

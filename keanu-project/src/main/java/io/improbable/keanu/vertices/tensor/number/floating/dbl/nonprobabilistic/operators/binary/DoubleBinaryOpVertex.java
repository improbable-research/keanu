package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;

public abstract class DoubleBinaryOpVertex
    extends VertexImpl<DoubleTensor, DoubleVertex>
    implements DoubleVertex, NonProbabilistic<DoubleTensor>, VertexBinaryOp<Vertex<DoubleTensor, ?>, Vertex<DoubleTensor, ?>> {

    protected final Vertex<DoubleTensor, ?> left;
    protected final Vertex<DoubleTensor, ?> right;
    protected static final String LEFT_NAME = "left";
    protected static final String RIGHT_NAME = "right";

    /**
     * A vertex that performs a user defined operation on two vertices
     *
     * @param left  a vertex
     * @param right a vertex
     */
    public DoubleBinaryOpVertex(Vertex<DoubleTensor, ?> left, Vertex<DoubleTensor, ?> right) {
        this(getBroadcastResultShape(left.getShape(), right.getShape()), left, right);
    }

    /**
     * A vertex that performs a user defined operation on two vertices
     *
     * @param shape the shape of the resulting vertex
     * @param left  a vertex
     * @param right a vertex
     */
    public DoubleBinaryOpVertex(long[] shape, Vertex<DoubleTensor, ?> left, Vertex<DoubleTensor, ?> right) {
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
    public Vertex<DoubleTensor, ?> getLeft() {
        return left;
    }

    @Override
    @SaveVertexParam(RIGHT_NAME)
    public Vertex<DoubleTensor, ?> getRight() {
        return right;
    }

    protected abstract DoubleTensor op(DoubleTensor l, DoubleTensor r);
}

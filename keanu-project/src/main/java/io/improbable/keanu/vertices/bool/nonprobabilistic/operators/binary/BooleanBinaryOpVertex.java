package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.BooleanVertex;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;

public abstract class BooleanBinaryOpVertex<A extends Tensor, B extends Tensor> extends VertexImpl<BooleanTensor> implements BooleanVertex, NonProbabilistic<BooleanTensor>, VertexBinaryOp<Vertex<A>, Vertex<B>> {

    protected final Vertex<A> left;
    protected final Vertex<B> right;
    protected final static String A_NAME = "left";
    protected final static String B_NAME = "right";

    public BooleanBinaryOpVertex(Vertex<A> left, Vertex<B> right) {
        this(getBroadcastResultShape(left.getShape(), right.getShape()), left, right);
    }

    public BooleanBinaryOpVertex(long[] shape, Vertex<A> left, Vertex<B> right) {
        super(shape);
        this.left = left;
        this.right = right;
        setParents(left, right);
    }

    @Override
    @SaveVertexParam(A_NAME)
    public Vertex<A> getLeft() {
        return left;
    }

    @Override
    @SaveVertexParam(B_NAME)
    public Vertex<B> getRight() {
        return right;
    }

    @Override
    public boolean contradictsObservation() {
        return isObserved() && !op(left.getValue(), right.getValue()).elementwiseEquals(getValue()).allTrue();
    }

    @Override
    public BooleanTensor calculate() {
        return op(left.getValue(), right.getValue());
    }

    protected abstract BooleanTensor op(A l, B r);

}
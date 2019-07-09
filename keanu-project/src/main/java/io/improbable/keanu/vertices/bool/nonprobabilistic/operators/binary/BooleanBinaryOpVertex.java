package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.BooleanVertex;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;

public abstract class BooleanBinaryOpVertex<A extends Tensor, B extends Tensor> extends VertexImpl<BooleanTensor> implements BooleanVertex, NonProbabilistic<BooleanTensor>, VertexBinaryOp<IVertex<A>, IVertex<B>> {

    protected final IVertex<A> left;
    protected final IVertex<B> right;
    protected final static String A_NAME = "left";
    protected final static String B_NAME = "right";

    public BooleanBinaryOpVertex(IVertex<A> left, IVertex<B> right) {
        this(getBroadcastResultShape(left.getShape(), right.getShape()), left, right);
    }

    public BooleanBinaryOpVertex(long[] shape, IVertex<A> left, IVertex<B> right) {
        super(shape);
        this.left = left;
        this.right = right;
        setParents(left, right);
    }

    @Override
    @SaveVertexParam(A_NAME)
    public IVertex<A> getLeft() {
        return left;
    }

    @Override
    @SaveVertexParam(B_NAME)
    public IVertex<B> getRight() {
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
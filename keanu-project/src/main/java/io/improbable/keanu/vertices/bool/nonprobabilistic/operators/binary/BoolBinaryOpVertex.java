package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public abstract class BoolBinaryOpVertex<A extends Tensor, B extends Tensor> extends BoolVertex implements NonProbabilistic<BooleanTensor>, VertexBinaryOp<Vertex<A>, Vertex<B>> {

    protected final Vertex<A> left;
    protected final Vertex<B> right;

    public BoolBinaryOpVertex(Vertex<A> left, Vertex<B> right) {
        this(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right);
    }

    public BoolBinaryOpVertex(long[] shape, Vertex<A> left, Vertex<B> right) {
        super(shape);
        this.left = left;
        this.right = right;
        setParents(left, right);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(left.sample(random), right.sample(random));
    }

    @Override
    public Vertex<A> getLeft() {
        return left;
    }

    @Override
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
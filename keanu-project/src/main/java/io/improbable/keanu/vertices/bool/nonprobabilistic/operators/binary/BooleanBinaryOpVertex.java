package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;

public abstract class BooleanBinaryOpVertex<A extends Tensor, B extends Tensor> extends BooleanVertex implements NonProbabilistic<BooleanTensor> {

    protected final Vertex<A> a;
    protected final Vertex<B> b;
    protected final static String A_NAME = "left";
    protected final static String B_NAME = "right";

    public BooleanBinaryOpVertex(Vertex<A> a, Vertex<B> b) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(a.getShape(), b.getShape()), a, b);
    }

    public BooleanBinaryOpVertex(long[] shape, Vertex<A> a, Vertex<B> b) {
        super(shape);
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(a.sample(random), b.sample(random));
    }

    @Override
    public boolean contradictsObservation() {
        return isObserved() && !op(a.getValue(), b.getValue()).elementwiseEquals(getValue()).allTrue();
    }

    @Override
    public BooleanTensor calculate() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract BooleanTensor op(A l, B r);

    @SaveVertexParam(A_NAME)
    public Vertex<A> getA() {
        return a;
    }

    @SaveVertexParam(B_NAME)
    public Vertex<B> getB() {
        return b;
    }
}
package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public abstract class BoolBinaryOpVertex<A extends Tensor, B extends Tensor> extends BoolVertex implements NonProbabilistic<BooleanTensor> {

    protected final Vertex<A> a;
    protected final Vertex<B> b;

    public BoolBinaryOpVertex(Vertex<A> a, Vertex<B> b) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b);
    }

    public BoolBinaryOpVertex(long[] shape, Vertex<A> a, Vertex<B> b) {
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
}
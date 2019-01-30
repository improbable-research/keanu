package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public abstract class BooleanUnaryOpVertex<T extends Tensor> extends BooleanVertex implements NonProbabilistic<BooleanTensor> {

    protected final Vertex<T> a;
    protected final static String INPUT_NAME = "inputVertex";

    public BooleanUnaryOpVertex(Vertex<T> a) {
        this(a.getShape(), a);
    }

    public BooleanUnaryOpVertex(long[] shape, Vertex<T> a) {
        super(shape);
        this.a = a;
        setParents(a);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(a.sample(random));
    }

    @Override
    public BooleanTensor calculate() {
        return op(a.getValue());
    }

    protected abstract BooleanTensor op(T value);

    @SaveVertexParam(INPUT_NAME)
    public Vertex<T> getA() {
        return a;
    }
}
package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public abstract class BoolUnaryOpVertex<T extends Tensor> extends BoolVertex implements NonProbabilistic<BooleanTensor> {

    protected final Vertex<T> a;
    protected final static String INPUT_NAME = "inputVertex";

    public BoolUnaryOpVertex(Vertex<T> a) {
        this(a.getShape(), a);
    }

    public BoolUnaryOpVertex(long[] shape, Vertex<T> a) {
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
package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public abstract class BooleanUnaryOpVertex<T extends Tensor> extends VertexImpl<BooleanTensor> implements BooleanVertex, NonProbabilistic<BooleanTensor>, VertexUnaryOp<IVertex<T>> {

    protected final IVertex<T> a;
    protected final static String INPUT_NAME = "inputVertex";

    public BooleanUnaryOpVertex(IVertex<T> a) {
        this(a.getShape(), a);
    }

    public BooleanUnaryOpVertex(long[] shape, IVertex<T> a) {
        super(shape);
        this.a = a;
        setParents(a);
    }

    @SaveVertexParam(INPUT_NAME)
    public IVertex<T> getInputVertex() {
        return a;
    }

    @Override
    public BooleanTensor calculate() {
        return op(a.getValue());
    }

    protected abstract BooleanTensor op(T value);

}
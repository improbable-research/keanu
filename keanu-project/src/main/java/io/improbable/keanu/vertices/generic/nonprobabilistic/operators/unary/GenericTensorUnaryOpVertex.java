package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

public abstract class GenericTensorUnaryOpVertex<IN, OUT> extends GenericTensorVertex<OUT> implements NonProbabilistic<GenericTensor<OUT>> {

    protected static final String INPUT_NAME = "inputVertex";

    protected final IVertex<GenericTensor<IN>> inputVertex;

    public GenericTensorUnaryOpVertex(long[] shape, IVertex<GenericTensor<IN>> inputVertex) {
        super(shape);
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public GenericTensor<OUT> calculate() {
        return op(inputVertex.getValue());
    }

    protected abstract GenericTensor<OUT> op(GenericTensor<IN> a);

    @SaveVertexParam(INPUT_NAME)
    public IVertex<GenericTensor<IN>> getInputVertex() {
        return inputVertex;
    }
}

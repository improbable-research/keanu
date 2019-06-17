package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

public abstract class GenericTensorUnaryOpVertex<IN, TENSOR_IN extends Tensor<IN, TENSOR_IN>, OUT, TENSOR_OUT extends Tensor<OUT, TENSOR_OUT>>
    extends GenericTensorVertex<OUT, TENSOR_OUT> implements NonProbabilistic<TENSOR_OUT> {

    protected static final String INPUT_NAME = "inputVertex";

    protected final Vertex<TENSOR_IN> inputVertex;

    public GenericTensorUnaryOpVertex(long[] shape, Vertex<TENSOR_IN> inputVertex) {
        super(shape);
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public TENSOR_OUT calculate() {
        return op(inputVertex.getValue());
    }

    protected abstract TENSOR_OUT op(TENSOR_IN a);

    @SaveVertexParam(INPUT_NAME)
    public Vertex<TENSOR_IN> getInputVertex() {
        return inputVertex;
    }
}

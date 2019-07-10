package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public abstract class DoubleUnaryOpVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, NonProbabilistic<DoubleTensor>, VertexUnaryOp<DoubleVertex> {

    protected final DoubleVertex inputVertex;
    protected static final String INPUT_VERTEX_NAME = "inputVertex";

    /**
     * A vertex that performs a user defined operation on a single input vertex
     *
     * @param inputVertex the input vertex
     */
    public DoubleUnaryOpVertex(DoubleVertex inputVertex) {
        this(inputVertex.getShape(), inputVertex);
    }

    /**
     * A vertex that performs a user defined operation on a single input vertex
     *
     * @param shape       the shape of the tensor
     * @param inputVertex the input vertex
     */
    public DoubleUnaryOpVertex(long[] shape, DoubleVertex inputVertex) {
        super(shape);
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @SaveVertexParam(INPUT_VERTEX_NAME)
    @Override
    public DoubleVertex getInputVertex() {
        return inputVertex;
    }

    @Override
    public DoubleTensor calculate() {
        return op(inputVertex.getValue());
    }

    protected abstract DoubleTensor op(DoubleTensor value);
}

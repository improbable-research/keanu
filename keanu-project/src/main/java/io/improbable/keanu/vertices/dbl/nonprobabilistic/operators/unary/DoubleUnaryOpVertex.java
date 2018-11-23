package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public abstract class DoubleUnaryOpVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor>, VertexUnaryOp<DoubleVertex> {

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

    @SaveParentVertex(INPUT_VERTEX_NAME)
    public DoubleVertex getInputVertex() {
        return inputVertex;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    @Override
    public DoubleTensor calculate() {
        return op(inputVertex.getValue());
    }

    public DoubleVertex getInput() {
        return inputVertex;
    }

    protected abstract DoubleTensor op(DoubleTensor value);
}

package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

public abstract class DoubleUnaryOpVertex
    extends VertexImpl<DoubleTensor, DoubleVertex>
    implements DoubleVertex, NonProbabilistic<DoubleTensor>, VertexUnaryOp<Vertex<DoubleTensor, ?>> {

    protected final Vertex<DoubleTensor, ?> inputVertex;
    protected static final String INPUT_NAME = "inputVertex";

    /**
     * A vertex that performs a user defined operation on a single input vertex
     *
     * @param inputVertex the input vertex
     */
    public DoubleUnaryOpVertex(Vertex<DoubleTensor, ?> inputVertex) {
        this(inputVertex.getShape(), inputVertex);
    }

    /**
     * A vertex that performs a user defined operation on a single input vertex
     *
     * @param shape       the shape of the tensor
     * @param inputVertex the input vertex
     */
    public DoubleUnaryOpVertex(long[] shape, Vertex<DoubleTensor, ?> inputVertex) {
        super(shape);
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @SaveVertexParam(INPUT_NAME)
    @Override
    public Vertex<DoubleTensor, ?> getInputVertex() {
        return inputVertex;
    }

    @Override
    public DoubleTensor calculate() {
        return op(inputVertex.getValue());
    }

    protected abstract DoubleTensor op(DoubleTensor value);
}

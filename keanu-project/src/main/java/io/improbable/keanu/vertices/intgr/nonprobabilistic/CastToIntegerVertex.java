package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class CastToIntegerVertex<T extends NumberTensor> extends Vertex<IntegerTensor> implements IntegerVertex, NonProbabilistic<IntegerTensor>, VertexUnaryOp<IVertex<T>> {

    private final IVertex<T> inputVertex;
    private static final String INPUT_NAME = "inputVertex";

    @ExportVertexToPythonBindings
    public CastToIntegerVertex(@LoadVertexParam(INPUT_NAME) IVertex<T> inputVertex) {
        super(inputVertex.getShape());
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public IntegerTensor calculate() {
        return inputVertex.getValue().toInteger();
    }

    @SaveVertexParam(INPUT_NAME)
    public IVertex<T> getInputVertex() {
        return inputVertex;
    }
}

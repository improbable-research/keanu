package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class BooleanToDoubleMaskVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, NonProbabilistic<DoubleTensor> {
    private static final String INPUT_NAME = "inputName";
    private final BooleanVertex inputVertex;

    @ExportVertexToPythonBindings
    public BooleanToDoubleMaskVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex inputVertex) {
        this.inputVertex = inputVertex;
    }

    @Override
    public DoubleTensor calculate() {
        return inputVertex.getValue().toDoubleMask();
    }

    @SaveVertexParam(INPUT_NAME)
    public BooleanVertex getInputVertex() {
        return this.inputVertex;
    }
}

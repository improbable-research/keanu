package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public class CastToBooleanVertex extends Vertex<BooleanTensor> implements BooleanVertex, NonProbabilistic<BooleanTensor> {

    private final IVertex<? extends BooleanTensor> inputVertex;
    private final static String INPUT_NAME = "inputVertex";

    @ExportVertexToPythonBindings
    public CastToBooleanVertex(@LoadVertexParam(INPUT_NAME) IVertex<? extends BooleanTensor> inputVertex) {
        super(inputVertex.getShape());
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public BooleanTensor calculate() {
        return inputVertex.getValue();
    }

    @SaveVertexParam(INPUT_NAME)
    public IVertex<? extends BooleanTensor> getInputVertex() {
        return inputVertex;
    }

}

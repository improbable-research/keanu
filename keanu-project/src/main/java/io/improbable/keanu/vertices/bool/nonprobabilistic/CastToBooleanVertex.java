package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public class CastToBooleanVertex extends BooleanVertex implements NonProbabilistic<BooleanTensor> {

    private final Vertex<? extends BooleanTensor> inputVertex;
    private final static String INPUT_NAME = "inputVertex";

    @ExportVertexToPythonBindings
    public CastToBooleanVertex(@LoadVertexParam(INPUT_NAME) Vertex<? extends BooleanTensor> inputVertex) {
        super(inputVertex.getShape());
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return inputVertex.sample(random);
    }

    @Override
    public BooleanTensor calculate() {
        return inputVertex.getValue();
    }

    @SaveVertexParam(INPUT_NAME)
    public Vertex<? extends BooleanTensor> getInputVertex() {
        return inputVertex;
    }

}

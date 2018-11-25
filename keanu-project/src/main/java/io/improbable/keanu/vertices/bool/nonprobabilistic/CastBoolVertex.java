package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class CastBoolVertex extends BoolVertex implements NonProbabilistic<BooleanTensor> {

    private final Vertex<? extends BooleanTensor> inputVertex;
    private final static String INPUT_NAME = "inputVertex";

    @ExportVertexToPythonBindings
    public CastBoolVertex(@LoadParentVertex(INPUT_NAME) Vertex<? extends BooleanTensor> inputVertex) {
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

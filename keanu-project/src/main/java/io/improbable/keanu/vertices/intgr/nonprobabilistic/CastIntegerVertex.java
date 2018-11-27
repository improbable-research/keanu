package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class CastIntegerVertex extends IntegerVertex implements NonProbabilistic<IntegerTensor> {

    private final Vertex<IntegerTensor> inputVertex;
    private static final String INPUT_NAME = "inputVertex";

    @ExportVertexToPythonBindings
    public CastIntegerVertex(@LoadParentVertex(INPUT_NAME) Vertex<IntegerTensor> inputVertex) {
        super(inputVertex.getShape());
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return inputVertex.sample(random);
    }

    @Override
    public IntegerTensor calculate() {
        return inputVertex.getValue();
    }

    @SaveParentVertex(INPUT_NAME)
    public Vertex<IntegerTensor> getInputVertex() {
        return inputVertex;
    }
}

package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

public class CastDoubleVertex extends DoubleVertex  implements NonProbabilistic<DoubleTensor> {

    private final Vertex<? extends NumberTensor> inputVertex;

    @ExportVertexToPythonBindings
    public CastDoubleVertex(Vertex<? extends NumberTensor> inputVertex) {
        super(inputVertex.getShape());
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return inputVertex.sample(random).toDouble();
    }

    @Override
    public DoubleTensor calculate() {
        return inputVertex.getValue().toDouble();
    }
}

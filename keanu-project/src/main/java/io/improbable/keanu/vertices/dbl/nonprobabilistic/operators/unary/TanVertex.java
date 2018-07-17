package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class TanVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the tangent of a vertex. Tan(vertex).
     *
     * @param inputVertex the vertex
     */
    public TanVertex(DoubleVertex inputVertex) {
        super(inputVertex, DoubleTensor::tan, DualNumber::tan);
    }
}

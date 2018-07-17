package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class CosVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the cosine of a vertex, Cos(vertex)
     *
     * @param inputVertex the vertex
     */
    public CosVertex(DoubleVertex inputVertex) {
        super(inputVertex, DoubleTensor::cos, DualNumber::cos);
    }
}

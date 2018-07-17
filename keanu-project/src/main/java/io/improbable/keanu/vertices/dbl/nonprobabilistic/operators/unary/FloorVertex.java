package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class FloorVertex extends DoubleUnaryOpVertex {

    /**
     * Applies the Floor operator to a vertex.
     * This maps a vertex to the biggest integer less than or equal to its value
     *
     * @param inputVertex the vertex to be floor'd
     */
    public FloorVertex(DoubleVertex inputVertex) {
        super(inputVertex, DoubleTensor::floor);
    }
}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class SinVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the sine of a vertex. Sin(vertex).
     *
     * @param inputVertex the vertex
     */
    public SinVertex(DoubleVertex inputVertex) {
        super(inputVertex, DoubleTensor::sin, DualNumber::sin);
    }
}

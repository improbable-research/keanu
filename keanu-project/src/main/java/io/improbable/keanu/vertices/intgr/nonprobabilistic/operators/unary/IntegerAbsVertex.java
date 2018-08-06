package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;


import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerAbsVertex extends IntegerUnaryOpVertex {

    /**
     * Takes the absolute value of a vertex
     * @param inputVertex the vertex
     */
    public IntegerAbsVertex(IntegerVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex, IntegerTensor::abs);
    }
}

package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;


import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SaveableVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerAbsVertex extends IntegerUnaryOpVertex implements SaveableVertex {

    /**
     * Takes the absolute value of a vertex
     * @param inputVertex the vertex
     */
    public IntegerAbsVertex(@LoadParentVertex(INPUT_NAME) IntegerVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        return value.abs();
    }
}

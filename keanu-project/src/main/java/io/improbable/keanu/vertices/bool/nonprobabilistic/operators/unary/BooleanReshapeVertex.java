package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;

/**
 * Returns the supplied vertex with a new shape of the same length
 **/
public class BooleanReshapeVertex extends BooleanUnaryOpVertex<BooleanTensor> {

    private final static String PROPOSED_SHAPE_NAME = "proposedShape";

    public BooleanReshapeVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> inputVertex,
                                @LoadVertexParam(PROPOSED_SHAPE_NAME) long... proposedShape) {
        super(proposedShape, inputVertex);
    }

    @Override
    protected BooleanTensor op(BooleanTensor value) {
        return value.reshape(getShape());
    }

    @SaveVertexParam(PROPOSED_SHAPE_NAME)
    public long[] getProposedShape() {
        return getShape();
    }
}

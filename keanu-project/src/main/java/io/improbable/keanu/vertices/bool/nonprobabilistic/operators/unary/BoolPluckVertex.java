package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerUnaryOpVertex;

public class BoolPluckVertex extends BoolUnaryOpVertex<BooleanTensor> {

    private final int[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex to extract from
     * @param index the index to extract at
     */
    public BoolPluckVertex(BoolVertex inputVertex, int... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
        this.index = index;
    }

    @Override
    protected BooleanTensor op(BooleanTensor a) {
        return BooleanTensor.scalar(a.getValue(index));
    }

}

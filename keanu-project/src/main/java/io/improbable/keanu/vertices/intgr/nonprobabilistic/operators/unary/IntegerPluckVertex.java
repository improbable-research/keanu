package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.Map;

public class IntegerPluckVertex extends IntegerUnaryOpVertex {

    private final int[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex to extract from
     * @param index the index to extract at
     */
    public IntegerPluckVertex(IntegerVertex inputVertex, int... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
        this.index = index;
    }

    @Override
    protected IntegerTensor op(IntegerTensor a) {
        return IntegerTensor.scalar(a.getValue(index));
    }

}

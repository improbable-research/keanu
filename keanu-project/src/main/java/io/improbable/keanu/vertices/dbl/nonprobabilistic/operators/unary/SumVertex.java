package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static java.util.Collections.singletonMap;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import java.util.Map;

public class SumVertex extends DoubleUnaryOpVertex {

    /**
     * Performs a sum across each value stored in a vertex
     *
     * @param inputVertex the vertex to have its values summed
     */
    public SumVertex(DoubleVertex inputVertex) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return DoubleTensor.scalar(value.sum());
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        return dualNumber.sum();
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(
            PartialDerivatives derivativeOfOutputsWithRespectToSelf) {

        PartialDerivatives derivativesWrtInput =
                derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(
                        DoubleTensor.ones(inputVertex.getShape()), this.getShape());

        return singletonMap(inputVertex, derivativesWrtInput);
    }
}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static java.util.Collections.singletonMap;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

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
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {

        Map<VertexId, DoubleTensor> derivativeOfOutWrtInput = new HashMap<>();
        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
            VertexId of = entry.getKey();
            DoubleTensor partialOfWrtSelf = entry.getValue();

            int[] outputShape = Arrays.copyOf(partialOfWrtSelf.getShape(), partialOfWrtSelf.getShape().length - getShape().length);
            DoubleTensor ones = DoubleTensor.ones(TensorShape.concat(outputShape, inputVertex.getShape()));
            DoubleTensor partial = ones.times(partialOfWrtSelf);
            derivativeOfOutWrtInput.put(of, partial);
        }

        return singletonMap(inputVertex, new PartialDerivatives(derivativeOfOutWrtInput));
    }
}

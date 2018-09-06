package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class TakeVertex extends DoubleUnaryOpVertex {

    private final int[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex to extract from
     * @param index       the index to extract at
     */
    public TakeVertex(DoubleVertex inputVertex, int... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
        this.index = index;
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return DoubleTensor.scalar(value.getValue(index));
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        return dualNumber.take(index);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> reshapedDerivatives = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partialDerivative : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
            DoubleTensor v = partialDerivative.getValue();
            int[] newPartialShape = updatePartialShapeToMatchInput(partialDerivative.getValue().getShape(), inputVertex.getShape());
            DoubleTensor zeros = DoubleTensor.zeros(newPartialShape);
            DoubleTensor stretched = zeros.plus(v);
            DoubleTensor mask = DoubleTensor.zeros(inputVertex.getShape());
            mask.setValue(1., index);
            DoubleTensor stretchedWithMask = stretched.times(mask);
            reshapedDerivatives.put(inputVertex, new PartialDerivatives(partialDerivative.getKey(), stretchedWithMask));
        }

        return reshapedDerivatives;
    }

    private int[] updatePartialShapeToMatchInput(int[] wrtSelfShape, int[] inputShape) {
        int[] partialShape = Arrays.copyOf(wrtSelfShape, wrtSelfShape.length);
        int ofLength = partialShape.length - inputShape.length;
        System.arraycopy(inputShape, 0, partialShape, ofLength, partialShape.length - ofLength);
        return partialShape;
    }
}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public class MultiplicationVertex extends DoubleBinaryOpVertex {

    /**
     * Multiplies one vertex by another
     *
     * @param left  vertex to be multiplied
     * @param right vertex to be multiplied
     */
    public MultiplicationVertex(DoubleVertex left, DoubleVertex right) {
        super(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right);
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        DualNumber leftDual = dualNumbers.get(left);
        DualNumber rightDual = dualNumbers.get(right);
        return leftDual.multiplyBy(rightDual);
    }

    @Override
    protected Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();

        PartialDerivatives rightPartial = derivativeOfOutputsWithRespectToSelf.multiplyBy(right.getValue());

        Map<Long, DoubleTensor> rightSummed = new HashMap<>();
        for (Map.Entry<Long, DoubleTensor> partialDerivative : rightPartial.asMap().entrySet()) {
            if (TensorShape.nonScalarDimensions(left.getShape()).length > 0) {
                rightSummed.put(partialDerivative.getKey(), partialDerivative.getValue());
            } else {
                int[] nonScalarDimensions = TensorShape.nonScalarDimensions(right.getShape()).length > 0 ? TensorShape.nonScalarDimensions(right.getShape()) : TensorShape.nonScalarDimensions(left.getShape());
                rightSummed.put(
                    partialDerivative.getKey(),
                    partialDerivative.getValue().sum(nonScalarDimensions).reshape(TensorShape.concat(right.getShape(), left.getShape())));
            }
        }
        partials.put(left, new PartialDerivatives(rightSummed));

        PartialDerivatives leftPartial = derivativeOfOutputsWithRespectToSelf.multiplyBy(left.getValue());

        Map<Long, DoubleTensor> leftSummed = new HashMap<>();
        for (Map.Entry<Long, DoubleTensor> partialDerivative : leftPartial.asMap().entrySet()) {
            if (TensorShape.nonScalarDimensions(right.getShape()).length > 0) {
                leftSummed.put(partialDerivative.getKey(), partialDerivative.getValue());
            } else {
                int[] nonScalarDimensions = TensorShape.nonScalarDimensions(right.getShape()).length > 0 ? TensorShape.nonScalarDimensions(right.getShape()) : TensorShape.nonScalarDimensions(left.getShape());
                leftSummed.put(
                    partialDerivative.getKey(),
                    partialDerivative.getValue().sum(nonScalarDimensions).reshape(TensorShape.concat(left.getShape(), right.getShape())));
            }
        }
        partials.put(right, new PartialDerivatives(leftSummed));
        return partials;
    }

    @Override
    protected DoubleTensor op(DoubleTensor left, DoubleTensor right) {
        return left.times(right);
    }
}

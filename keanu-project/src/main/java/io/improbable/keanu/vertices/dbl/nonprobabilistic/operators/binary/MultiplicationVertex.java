package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

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
        Map<Long, DoubleTensor> toScalar = new HashMap<>();

        partials.put(left, derivativeOfOutputsWithRespectToSelf.multiplyBy(right.getValue()));

        if (right.getValue().isScalar()) {
            for (Map.Entry<Long, DoubleTensor> partialDerivative : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
                //replace ones with diag of matrix?
                toScalar.put(
                    partialDerivative.getKey(),
                    DoubleTensor.ones(
                        left.getShape()
                    ).reshape(TensorShape.shapeDesiredToRankByAppendingOnes(left.getShape(), partialDerivative.getValue().getRank()))
                );
            }
            partials.put(right, new PartialDerivatives(toScalar).multiplyBy(left.getValue()));
        } else {
            partials.put(right, derivativeOfOutputsWithRespectToSelf.multiplyBy(left.getValue()));
        }
        return partials;
    }

    @Override
    protected DoubleTensor op(DoubleTensor left, DoubleTensor right) {
        return left.times(right);
    }
}

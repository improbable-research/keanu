package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

public class MultiplicationVertex extends DoubleBinaryOpVertex {

    /**
     * Multiplies one vertex by another
     *
     * @param left  vertex to be multiplied
     * @param right vertex to be multiplied
     */
    @ExportVertexToPythonBindings
    public MultiplicationVertex(DoubleVertex left, DoubleVertex right) {
        super(left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.times(r);
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives dLeftWrtInputs, PartialDerivatives dRightWrtInputs) {

        // dc = A * db + da * B;
        PartialDerivatives partialsFromLeft;
        PartialDerivatives partialsFromRight;

        if (dLeftWrtInputs.isEmpty()) {
            partialsFromLeft = PartialDerivatives.OF_CONSTANT;
        } else {
            partialsFromLeft = dLeftWrtInputs.multiplyAlongOfDimensions(right.getValue(), left.getValue().getShape());
        }

        if (dRightWrtInputs.isEmpty()) {
            partialsFromRight = PartialDerivatives.OF_CONSTANT;
        } else {
            partialsFromRight = dRightWrtInputs.multiplyAlongOfDimensions(left.getValue(), right.getValue().getShape());
        }

        return partialsFromLeft.add(partialsFromRight);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();

        PartialDerivatives dOutputsWrtLeft = derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(right.getValue(), this.getShape());
        PartialDerivatives dOutputsWrtRight = derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(left.getValue(), this.getShape());

        partials.put(left, dOutputsWrtLeft);
        partials.put(right, dOutputsWrtRight);

        return partials;
    }
}

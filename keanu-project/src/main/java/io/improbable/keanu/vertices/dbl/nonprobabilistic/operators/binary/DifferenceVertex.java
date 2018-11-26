package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

@DisplayInformationForOutput(displayName = "-")
public class DifferenceVertex extends DoubleBinaryOpVertex {

    /**
     * Subtracts one vertex from another
     *
     * @param left  the vertex that will be subtracted from
     * @param right the vertex to subtract
     */
    @ExportVertexToPythonBindings
    public DifferenceVertex(@LoadVertexParam(LEFT_NAME) DoubleVertex left,
                            @LoadVertexParam(RIGHT_NAME) DoubleVertex right) {
        super(left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.minus(r);
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives dLeftWrtInputs, PartialDerivatives dRightWrtInputs) {
        return dLeftWrtInputs.subtract(dRightWrtInputs, getShape());
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(left, derivativeOfOutputsWithRespectToSelf);
        partials.put(right, derivativeOfOutputsWithRespectToSelf.multiplyBy(-1.0));
        return partials;
    }
}

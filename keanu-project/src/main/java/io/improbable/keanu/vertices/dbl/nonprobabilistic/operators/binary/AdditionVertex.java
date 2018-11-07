package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

public class AdditionVertex extends DoubleBinaryOpVertex {

    /**
     * Adds one vertex to another
     *
     * @param left  a vertex to add
     * @param right a vertex to add
     */
    @ExportVertexToPythonBindings
    public AdditionVertex(DoubleVertex left, DoubleVertex right) {
        super(left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.plus(r);
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives dLeftWrtInputs, PartialDerivatives dRightWrtInputs) {
        return dLeftWrtInputs.add(dRightWrtInputs,
            TensorShape.isScalar(left.getShape()),
            TensorShape.isScalar(right.getShape()),
            this.getShape()
        );
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(left, derivativeOfOutputsWithRespectToSelf);
        partials.put(right, derivativeOfOutputsWithRespectToSelf);
        return partials;
    }

}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class AdditionVertex extends DoubleBinaryOpVertex {

    /**
     * Adds one vertex to another
     *
     * @param left  a vertex to add
     * @param right a vertex to add
     */
    public AdditionVertex(DoubleVertex left, DoubleVertex right) {
        super(left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.plus(r);
    }

    @Override
    protected PartialDerivatives dualOp(PartialDerivatives l, PartialDerivatives r) {
        return l.add(r, this.getShape());
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(left, derivativeOfOutputsWithRespectToSelf);
        partials.put(right, derivativeOfOutputsWithRespectToSelf);
        return partials;
    }

}

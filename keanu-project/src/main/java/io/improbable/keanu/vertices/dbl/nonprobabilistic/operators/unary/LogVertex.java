package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

public class LogVertex extends DoubleUnaryOpVertex {

    /**
     * Returns the natural logarithm, base e, of a vertex
     *
     * @param inputVertex the vertex
     */
    public LogVertex(DoubleVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    protected DoubleTensor op(DoubleTensor a) {
        return a.log();
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return dualNumbers.get(inputVertex).log();
    }

    @Override
    protected Map<Vertex, PartialDerivatives> derivativeWithRespectTo(PartialDerivatives dAlldSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(inputVertex, dAlldSelf.multiplyBy(inputVertex.getValue().reciprocal()));
        return partials;
    }
}

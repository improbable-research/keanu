package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;

public class PowerVertex extends DoubleBinaryOpVertex implements Differentiable {

    /**
     * Raises a vertex to the power of another
     *
     * @param a the base vertex
     * @param b the exponent vertex
     */
    public PowerVertex(DoubleVertex a, DoubleVertex b) {
        super(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a, DoubleTensor b) {
        return a.pow(b);
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        DualNumber aDual = dualNumbers.get(a);
        DualNumber bDual = dualNumbers.get(b);
        return aDual.pow(bDual);
    }
}

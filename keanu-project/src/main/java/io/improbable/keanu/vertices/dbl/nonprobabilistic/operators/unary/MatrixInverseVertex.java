package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

public class MatrixInverseVertex extends DoubleUnaryOpVertex implements Differentiable {

    public MatrixInverseVertex(DoubleVertex inputVertex) {
        super(checkInputIsSquareMatrix(inputVertex.getShape()), inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.matrixInverse();
    }

    @Override
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        PartialDerivatives derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInputs.get(inputVertex);

        //dc = -A^-1 * da * A^-1
        DoubleTensor negatedValue = this.getValue().unaryMinus();
        PartialDerivatives partial = PartialDerivatives.matrixMultiplyAlongOfDimensions(derivativeOfParentWithRespectToInputs, negatedValue, false);
        partial = PartialDerivatives.matrixMultiplyAlongOfDimensions(partial, this.getValue(), true);
        return partial;
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        DoubleTensor parentValue = getValue();
        DoubleTensor negativeValue = getValue().unaryMinus();

        PartialDerivatives newPartials =
            PartialDerivatives.matrixMultiplyAlongWrtDimensions(derivativeOfOutputsWithRespectToSelf, negativeValue, false);
        newPartials = PartialDerivatives.matrixMultiplyAlongWrtDimensions(newPartials, parentValue, true);

        partials.put(inputVertex, newPartials);

        return partials;
    }

    private static long[] checkInputIsSquareMatrix(long[] shape) {
        if (shape.length != 2) {
            throw new IllegalArgumentException("Can only invert a Matrix (received rank: " + shape.length + ")");
        }

        if (shape[0] != shape[1]) {
            throw new IllegalArgumentException("Can only invert a square Matrix (received: "
                + shape[0] + ", " + shape[1] + ")");
        }

        return shape;
    }

}

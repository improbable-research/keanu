package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ReshapeVertex extends DoubleUnaryOpVertex {

    public ReshapeVertex(DoubleVertex inputVertex, int... proposedShape) {
        super(proposedShape, inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.reshape(getShape());
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives derivativeOfParentWithRespectToInputs) {
        return derivativeOfParentWithRespectToInputs.reshape(inputVertex.getValue().getRank(), getShape());
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> reshapedDerivatives = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partialDerivative : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
            DoubleTensor partial = partialDerivative.getValue();
            int[] newPartialShape = TensorShape.concat(
                TensorShape.selectDimensions(0, partial.getRank() - getShape().length, partial.getShape()),
                inputVertex.getShape()
            );
            DoubleTensor reshapedPartialDerivative = partialDerivative.getValue().reshape(newPartialShape);
            reshapedDerivatives.put(inputVertex, new PartialDerivatives(partialDerivative.getKey(), reshapedPartialDerivative));
        }

        return reshapedDerivatives;
    }

}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

public class ReshapeVertex extends DoubleUnaryOpVertex implements Differentiable, NonSaveableVertex {

    @ExportVertexToPythonBindings
    public ReshapeVertex(DoubleVertex inputVertex, long... proposedShape) {
        super(proposedShape, inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.reshape(getShape());
    }

    @Override
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        PartialDerivatives derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInputs.get(inputVertex);
        return derivativeOfParentWithRespectToInputs.reshape(inputVertex.getValue().getRank(), getShape());
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> reshapedDerivatives = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partialDerivative : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
            DoubleTensor partial = partialDerivative.getValue();
            long[] newPartialShape = TensorShape.concat(
                TensorShape.selectDimensions(0, partial.getRank() - getShape().length, partial.getShape()),
                inputVertex.getShape()
            );
            DoubleTensor reshapedPartialDerivative = partialDerivative.getValue().reshape(newPartialShape);
            reshapedDerivatives.put(inputVertex, new PartialDerivatives(partialDerivative.getKey(), reshapedPartialDerivative));
        }

        return reshapedDerivatives;
    }

}

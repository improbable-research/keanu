package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import io.improbable.keanu.vertices.number.UnaryTensorOpVertex;

import java.util.HashMap;
import java.util.Map;

public class ReshapeVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String SHAPE_NAME = "proposedShape";

    @ExportVertexToPythonBindings
    public ReshapeVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                         @LoadVertexParam(SHAPE_NAME) long... proposedShape) {
        super(proposedShape, inputVertex);
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.reshape(getShape());
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative dInputVertex = derivativeOfParentsWithRespectToInput.get(inputVertex);

        long[] newPartialShape = TensorShape.concat(
            getShape(),
            dInputVertex.getWrtShape(inputVertex.getShape())
        );

        return new PartialDerivative(dInputVertex.get().reshape(newPartialShape));
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> reshapedDerivatives = new HashMap<>();

        long[] newPartialShape = TensorShape.concat(
            derivativeOfOutputWithRespectToSelf.getOfShape(getShape()),
            inputVertex.getShape()
        );

        PartialDerivative dXWrtInputVertex = new PartialDerivative(
            derivativeOfOutputWithRespectToSelf.get().reshape(newPartialShape)
        );

        reshapedDerivatives.put(inputVertex, dXWrtInputVertex);

        return reshapedDerivatives;
    }

    @SaveVertexParam(SHAPE_NAME)
    public long[] getProposedShape() {
        return getShape();
    }
}

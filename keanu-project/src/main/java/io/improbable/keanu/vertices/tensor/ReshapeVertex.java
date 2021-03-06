package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class ReshapeVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String SHAPE_NAME = "proposedShape";

    @ExportVertexToPythonBindings
    public ReshapeVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                         @LoadVertexParam(SHAPE_NAME) long... proposedShape) {
        super(proposedShape, inputVertex, inputVertex.ofType());
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.reshape(getShape());
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        ForwardModePartialDerivative dInputVertex = derivativeOfParentsWithRespectToInput.get(inputVertex);

        long[] newPartialShape = TensorShape.concat(
            dInputVertex.getWrtShape(),
            getShape()
        );

        return new ForwardModePartialDerivative(dInputVertex.getWrtShape(), dInputVertex.get().reshape(newPartialShape));
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, ReverseModePartialDerivative> reshapedDerivatives = new HashMap<>();

        long[] ofShape = derivativeOfOutputWithRespectToSelf.getOfShape();
        long[] newPartialShape = TensorShape.concat(
            ofShape,
            inputVertex.getShape()
        );

        ReverseModePartialDerivative dXWrtInputVertex = new ReverseModePartialDerivative(
            ofShape,
            derivativeOfOutputWithRespectToSelf.get().reshape(newPartialShape)
        );

        reshapedDerivatives.put(inputVertex, dXWrtInputVertex);

        return reshapedDerivatives;
    }

    @SaveVertexParam(SHAPE_NAME)
    public long[] getProposedShape() {
        return getShape();
    }

    @Override
    public boolean isDifferentiable() {
        return inputVertex.isDifferentiable();
    }

}

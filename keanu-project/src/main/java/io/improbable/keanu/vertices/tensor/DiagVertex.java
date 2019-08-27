package io.improbable.keanu.vertices.tensor;

import com.google.common.base.Preconditions;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Collections;
import java.util.Map;

public class DiagVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    @ExportVertexToPythonBindings
    public DiagVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(getDiagShape(inputVertex.getShape()), inputVertex, inputVertex.ofType());
    }

    private static long[] getDiagShape(long[] inputShape) {
        Preconditions.checkArgument(inputShape.length > 0, "Diag only operates on rank >= 1");
        return TensorShape.getDiagResultShape(inputShape);
    }

    @Override
    protected TENSOR op(TENSOR input) {
        return input.diag();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        final PartialDerivative partial = derivativeOfParentsWithRespectToInput.get(inputVertex);
        return PartialDerivative.createFromWrtOf(partial.getWrtOf(inputVertex.getRank()).diag(), this.getRank());
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {
        return Collections.singletonMap(inputVertex, new PartialDerivative(partial.get().diagPart()));
    }
}

package io.improbable.keanu.vertices.tensor;

import com.google.common.base.Preconditions;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Collections;
import java.util.Map;

public class TriLowerVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String K_NAME = "k";
    private final int k;

    @ExportVertexToPythonBindings
    public TriLowerVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                          @LoadVertexParam(K_NAME) int k) {
        super(getTriShape(inputVertex.getShape()), inputVertex, inputVertex.ofType());
        this.k = k;
    }

    private static long[] getTriShape(long[] inputShape) {
        Preconditions.checkArgument(inputShape.length >= 2, "Tri Upper only operates on rank >= 2");
        return inputShape;
    }

    @Override
    protected TENSOR op(TENSOR input) {
        return input.triLower(k);
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        final ForwardModePartialDerivative partial = derivativeOfParentsWithRespectToInput.get(inputVertex);
        return new ForwardModePartialDerivative(partial.getWrtShape(), partial.get().triLower(k));
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {
        return Collections.singletonMap(inputVertex, new PartialDerivative(partial.getOfShape(), partial.get().triLower(k)));
    }

    @SaveVertexParam(K_NAME)
    public int getK() {
        return k;
    }
}

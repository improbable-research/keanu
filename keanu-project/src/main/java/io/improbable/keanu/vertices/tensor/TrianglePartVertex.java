package io.improbable.keanu.vertices.tensor;

import com.google.common.base.Preconditions;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Collections;
import java.util.Map;

public class TrianglePartVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String UPPER_PART_NAME = "upperPart";
    private final boolean upperPart;

    @ExportVertexToPythonBindings
    public TrianglePartVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                              @LoadVertexParam(UPPER_PART_NAME) boolean upperPart) {
        super(trianglePartShape(inputVertex.getShape()), inputVertex, inputVertex.ofType());
        this.upperPart = upperPart;
    }

    private static long[] trianglePartShape(long[] shape) {
        Preconditions.checkArgument(shape.length >= 2, "Triangular part only operates on rank >= 2");

        final long N = shape[shape.length - 2];
        final long M = shape[shape.length - 1];

        Preconditions.checkArgument(
            N == M,
            "Triangle part only operates on square matrices. Received " + M + "," + N + "."
        );

        final long length = N * (N + 1) / 2;
        return TensorShape.concat(ArrayUtils.subarray(shape, 0, shape.length - 2), new long[]{length});
    }

    @Override
    protected TENSOR op(TENSOR input) {
        return input.trianglePart(upperPart);
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        final ForwardModePartialDerivative partial = derivativeOfParentsWithRespectToInput.get(inputVertex);
        return new ForwardModePartialDerivative(partial.getWrtShape(), partial.get().trianglePart(upperPart));
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {
        final DoubleTensor result;
        if (upperPart) {
            result = partial.get().fillTriangular(true, false);
        } else {
            result = partial.get().fillTriangular(false, true);
        }

        return Collections.singletonMap(inputVertex, new PartialDerivative(partial.getOfShape(), result));
    }

    @SaveVertexParam(UPPER_PART_NAME)
    public boolean isUpperPart() {
        return upperPart;
    }

}

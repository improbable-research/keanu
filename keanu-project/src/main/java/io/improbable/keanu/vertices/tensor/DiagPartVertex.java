package io.improbable.keanu.vertices.tensor;

import com.google.common.base.Preconditions;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;
import org.apache.commons.lang3.ArrayUtils;

import java.util.HashMap;
import java.util.Map;

public class DiagPartVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    @ExportVertexToPythonBindings
    public DiagPartVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(getDiagPartShape(inputVertex.getShape()), inputVertex, inputVertex.ofType());
    }

    private static long[] getDiagPartShape(long[] inputShape) {
        Preconditions.checkArgument(inputShape.length >= 2, "Diag Part operates on matrices or greater rank");
        return TensorShape.getDiagPartResultShape(inputShape);
    }

    @Override
    protected TENSOR op(TENSOR input) {
        return input.diagPart();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {

        PartialDerivative partial = derivativeOfParentsWithRespectToInput.get(inputVertex);
        final long[] inputShape = inputVertex.getShape();
        final long[] wrtShape = partial.getWrtShape(inputShape);

        final long M = inputShape[inputShape.length - 2];
        final long N = inputShape[inputShape.length - 1];
        final long inputMatrixLength = M * N;
        final long resultLength = Math.min(M, N);
        final long[] flatMatrixShape = ArrayUtils.subarray(inputShape, 0, inputShape.length - 1);
        flatMatrixShape[flatMatrixShape.length - 1] = inputMatrixLength;

        Slicer.SlicerBuilder slicerBuilder = Slicer.builder();

        for (int i = 0; i < inputShape.length - 2; i++) {
            slicerBuilder.all();
        }

        Slicer slicer = slicerBuilder
            .slice(0L, inputMatrixLength, resultLength + 1L)
            .build();

        DoubleTensor result = partial.get().reshape(TensorShape.concat(flatMatrixShape, wrtShape)).slice(slicer);

        return new PartialDerivative(result);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {
        HashMap<Vertex, PartialDerivative> result = new HashMap<>();
        result.put(inputVertex, new PartialDerivative(partial.get().diag()));
        return result;
    }
}

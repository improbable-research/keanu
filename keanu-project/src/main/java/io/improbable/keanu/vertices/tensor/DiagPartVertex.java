package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
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
        return new long[]{Math.min(inputShape[0], inputShape[1])};
    }

    @Override
    protected TENSOR op(TENSOR input) {
        return input.diagPart();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {

        PartialDerivative partial = derivativeOfParentsWithRespectToInput.get(inputVertex);

        final long[] inputShape = inputVertex.getShape();
        final long inputLength = inputVertex.getLength();

        final long[] wrtShape = partial.getWrtShape(inputShape);
        final long resultLength = Math.min(inputShape[0], inputShape[1]);

        final long[] ofFlattenedShape = ArrayUtils.insert(0, wrtShape, inputLength);

        DoubleTensor result = partial.get().reshape(ofFlattenedShape)
            .slice(Slicer.builder()
                .slice(0L, inputLength, resultLength + 1L)
                .build()
            );

        return new PartialDerivative(result);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {
        HashMap<Vertex, PartialDerivative> result = new HashMap<>();
        result.put(inputVertex, new PartialDerivative(partial.get().diag()));
        return result;
    }
}

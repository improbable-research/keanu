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
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import org.apache.commons.lang3.ArrayUtils;

import java.util.HashMap;
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
        PartialDerivative partial = derivativeOfParentsWithRespectToInput.get(inputVertex);

        final long[] ofShape = inputVertex.getShape();
        final long[] wrtShape = partial.getWrtShape(ofShape);
        final long N = ofShape[ofShape.length - 1];
        final long wrtShapeLength = TensorShape.getLength(wrtShape);
        final long[] resultShape = TensorShape.concat(getShape(), wrtShape);
        final DoubleTensor zeroes = DoubleTensor.zeros(resultShape).reshape(-1, N * N, wrtShapeLength);

        DoubleTensor result = partial.get().reverseSlice(
            zeroes,
            Slicer.builder()
                .all()
                .slice(0L, null, N + 1L)
                .build()
        ).reshape(resultShape);

        return new PartialDerivative(result);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {

        final DoubleTensor p = partial.get();
        final long[] pShape = p.getShape();

        final long endDim = pShape[pShape.length - 1];
        final long endDimsLength = endDim * endDim;

        final long[] firstDims = ArrayUtils.subarray(pShape, 0, p.getRank() - 1);
        firstDims[firstDims.length - 1] = endDimsLength;

        final DoubleTensor inputResultPartial = partial.get()
            .reshape(firstDims)
            .slice(Slicer.builder()
                .ellipsis()
                .slice(0L, endDimsLength, endDim + 1L)
                .build()
            );

        final HashMap<Vertex, PartialDerivative> result = new HashMap<>();
        result.put(inputVertex, new PartialDerivative(inputResultPartial));
        return result;
    }
}

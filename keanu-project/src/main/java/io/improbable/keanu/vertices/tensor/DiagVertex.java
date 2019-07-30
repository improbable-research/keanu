package io.improbable.keanu.vertices.tensor;

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
        return TensorShape.concat(inputShape, new long[]{inputShape[inputShape.length - 1]});
    }

    @Override
    protected TENSOR op(TENSOR input) {
        return input.diag();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {

        final DoubleTensor p = partial.get();
        final long[] pShape = p.getShape();

        final long endDim = pShape[pShape.length - 1];
        final long endDimsLength = endDim * endDim;

        final long[] firstDims = ArrayUtils.subarray(pShape, 0, p.getRank() - 1);
        firstDims[firstDims.length - 1] = endDimsLength;

        Slicer.SlicerBuilder builder = Slicer.builder();

        for (int i = 0; i < firstDims.length - 1; i++) {
            builder.all();
        }

        Slicer slicer = builder.slice(0L, endDimsLength, endDim + 1L).build();

        DoubleTensor inputResultPartial = partial.get().reshape(firstDims).slice(slicer);

        HashMap<Vertex, PartialDerivative> result = new HashMap<>();
        result.put(inputVertex, new PartialDerivative(inputResultPartial));
        return result;
    }
}

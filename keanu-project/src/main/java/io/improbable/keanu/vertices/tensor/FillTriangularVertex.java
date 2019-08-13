package io.improbable.keanu.vertices.tensor;

import com.google.common.base.Preconditions;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Map;

public class FillTriangularVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String FILL_UPPER_NAME = "fillUpper";
    private static final String FILL_LOWER_NAME = "fillLower";

    private final boolean fillUpper;
    private final boolean fillLower;

    @ExportVertexToPythonBindings
    public FillTriangularVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                                @LoadVertexParam(FILL_UPPER_NAME) boolean fillUpper,
                                @LoadVertexParam(FILL_LOWER_NAME) boolean fillLower) {
        super(fillTriangularShape(inputVertex.getShape()), inputVertex, inputVertex.ofType());
        this.fillUpper = fillUpper;
        this.fillLower = fillLower;
    }

    private static long[] fillTriangularShape(long[] shape) {
        Preconditions.checkArgument(shape.length >= 1, "Fill triangular only operates on rank >= 1");

        final long endDim = shape[shape.length - 1];
        double a = Math.sqrt(1 + 8 * endDim);

        Preconditions.checkArgument(
            a == Math.floor(a),
            "Length " + endDim + " is not the correct number of elements for a triangular matrix"
        );

        final long N = ((long) a - 1) / 2;
        return TensorShape.concat(ArrayUtils.subarray(shape, 0, shape.length - 1), new long[]{N, N});
    }

    @Override
    protected TENSOR op(TENSOR input) {
        return input.fillTriangular(fillUpper, fillLower);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        final PartialDerivative partial = derivativeOfParentsWithRespectToInput.get(inputVertex);
        return PartialDerivative.createFromWrtOf(partial.getWrtOf(inputVertex.getRank()).fillTriangular(fillUpper, fillLower), this.getRank());
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {

        return null;
    }

    @SaveVertexParam(FILL_UPPER_NAME)
    public boolean isFillUpper() {
        return fillUpper;
    }

    @SaveVertexParam(FILL_LOWER_NAME)
    public boolean isFillLower() {
        return fillLower;
    }
}

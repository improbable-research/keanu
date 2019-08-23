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
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        final ForwardModePartialDerivative partial = derivativeOfParentsWithRespectToInput.get(inputVertex);
        return new ForwardModePartialDerivative(partial.getWrtShape(), partial.get().fillTriangular(fillUpper, fillLower));
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {

        final DoubleTensor p = partial.get();
        if (fillUpper && fillLower) {
            final PartialDerivative result = new PartialDerivative(partial.getOfShape(), p.trianglePart(true).plus(p.triLower(1).trianglePart(false)));
            return Collections.singletonMap(inputVertex, result);
        } else if (fillLower) {
            final PartialDerivative result = new PartialDerivative(partial.getOfShape(), p.trianglePart(false));
            return Collections.singletonMap(inputVertex, result);
        } else if (fillUpper) {
            final PartialDerivative result = new PartialDerivative(partial.getOfShape(), p.trianglePart(true));
            return Collections.singletonMap(inputVertex, result);
        } else {
            final long[] ofShape = partial.getOfShape();
            final PartialDerivative result = new PartialDerivative(partial.getOfShape(), DoubleTensor.zeros(TensorShape.concat(ofShape, inputVertex.getShape())));
            return Collections.singletonMap(inputVertex, result);
        }
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

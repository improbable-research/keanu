package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Arrays;

/**
 * This class is meant to help with auto diff in operations that support implicit broadcasting. E.g. In
 * addition/subtraction/multiplication/division scalar operands can be operated with non-scalar operands.
 */
public class AutoDiffBroadcast {

    public static PartialDerivative correctForScalarPartialForward(PartialDerivative partial, long[] partialOfShape, long[] targetOfShape) {

        if (shouldCorrectPartialForScalarForward(partial, partialOfShape, targetOfShape)) {

            long[] wrtShape = partial.getWrtShape(partialOfShape);
            DoubleTensor correctedPartial = DoubleTensor
                .zeros(TensorShape.concat(targetOfShape, wrtShape))
                .plus(partial.getPartial());

            return new PartialDerivative(correctedPartial);
        } else {
            return partial;
        }
    }

    private static boolean shouldCorrectPartialForScalarForward(PartialDerivative partial, long[] partialOfShape, long[] targetOfShape) {
        return partial.isPresent() && !Arrays.equals(partialOfShape, targetOfShape);
    }

    public static PartialDerivative correctForScalarPartialReverse(PartialDerivative partial, long[] partialWrtShape, long[] targetWrtShape) {

        if (shouldCorrectForPartialScalarReverse(partial, partialWrtShape, targetWrtShape)) {

            int[] wrtDims = TensorShape.dimensionRange(-partialWrtShape.length, 0);
            DoubleTensor partialSummed = partial.getPartial().sum(wrtDims);

            long[] resultShape = TensorShape.concat(
                partial.getOfShape(partialWrtShape),
                targetWrtShape
            );

            return new PartialDerivative(partialSummed.reshape(resultShape));
        } else {
            return partial;
        }
    }

    public static boolean shouldCorrectForPartialScalarReverse(PartialDerivative partial, long[] partialWrtShape, long[] targetWrtShape) {
        return partial.isPresent() && !Arrays.equals(partialWrtShape, targetWrtShape);
    }
}

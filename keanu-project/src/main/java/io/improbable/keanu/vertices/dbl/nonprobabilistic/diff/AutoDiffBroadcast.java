package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.experimental.UtilityClass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class is meant to help with auto diff in operations that support implicit broadcasting. E.g. In
 * addition/subtraction/multiplication/division scalar operands can be operated with non-scalar operands.
 */
@UtilityClass
public class AutoDiffBroadcast {

    public static PartialDerivative correctForBroadcastPartialForward(PartialDerivative partial, long[] partialOfShape, long[] targetOfShape) {

        if (shouldCorrectPartialForBroadcast(partial, partialOfShape, targetOfShape)) {

            long[] wrtShape = partial.getWrtShape(partialOfShape);
            DoubleTensor correctedPartial = DoubleTensor
                .zeros(TensorShape.concat(targetOfShape, wrtShape))
                .plus(partial.get());

            return new PartialDerivative(correctedPartial);
        } else {
            return partial;
        }
    }

    public static PartialDerivative correctForBroadcastPartialReverse(PartialDerivative partial, long[] partialWrtShape, long[] targetWrtShape) {

        if (shouldCorrectPartialForBroadcast(partial, partialWrtShape, targetWrtShape)) {

            long[] partialShape = partial.get().getShape();

            int[] broadcastDimensions = dimensionsWithShapeChange(partialShape, partialWrtShape.length, targetWrtShape);

            DoubleTensor partialSummed = partial.get().sum(broadcastDimensions);

            long[] resultShape = TensorShape.concat(
                partial.getOfShape(partialWrtShape),
                targetWrtShape
            );

            return new PartialDerivative(partialSummed.reshape(resultShape));
        } else {
            return partial;
        }
    }

    /**
     * @param partial       The partial derivative that may or may not come from a broadcasted operation.
     * @param actualShape   The part of the partial shape that should match the expected shape in the case no broadcast was
     *                      performed. This would be the of shape for forward mode and the with respect to shape for reverse.
     * @param expectedShape The shape of the operation result in forward mode or the shape of the operand in reverse mode.
     *                      This should match the actual shape if no broadcast was performed.
     * @return true if a broadcast should be taken into account and corrected for in the auto diff calculation, false otherwise.
     */
    private static boolean shouldCorrectPartialForBroadcast(PartialDerivative partial, long[] actualShape, long[] expectedShape) {
        return partial.isPresent() && !Arrays.equals(actualShape, expectedShape);
    }

    private static int[] dimensionsWithShapeChange(long[] partialShape, int partialWrtRank, long[] wrtShape) {

        final int partialRank = partialShape.length;
        final int wrtRank = wrtShape.length;
        List<Integer> dimensionMismatch = new ArrayList<>();

        for (int i = 1; i <= partialWrtRank; i++) {
            if (i > wrtRank || partialShape[partialRank - i] != wrtShape[wrtRank - i]) {
                dimensionMismatch.add(-i);
            }
        }

        return Ints.toArray(dimensionMismatch);
    }
}

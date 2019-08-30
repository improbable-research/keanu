package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff;

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

    public static ReverseModePartialDerivative correctForBroadcastPartialReverse(ReverseModePartialDerivative partial, long[] partialWrtShape, long[] targetWrtShape) {

        if (shouldCorrectPartialForBroadcast(partial, partialWrtShape, targetWrtShape)) {
            return broadcastPartialReverse(partial, partialWrtShape, targetWrtShape);
        } else {
            return partial;
        }
    }

    public static ReverseModePartialDerivative broadcastPartialReverse(ReverseModePartialDerivative partial, long[] partialWrtShape, long[] targetWrtShape) {
        long[] partialShape = partial.get().getShape();

        int[] broadcastDimensions = dimensionsWithShapeChange(partialShape, partialWrtShape.length, targetWrtShape);

        DoubleTensor partialSummed = partial.get().sum(broadcastDimensions);

        long[] resultShape = TensorShape.concat(
            partial.getOfShape(),
            targetWrtShape
        );

        return new ReverseModePartialDerivative(partial.getOfShape(), partialSummed.reshape(resultShape));
    }

    private static boolean shouldCorrectPartialForBroadcast(ReverseModePartialDerivative partial, long[] actualShape, long[] expectedShape) {
        return partial.isPresent() && !Arrays.equals(actualShape, expectedShape);
    }

    public static int[] dimensionsWithShapeChange(long[] partialShape, int partialWrtRank, long[] wrtShape) {

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

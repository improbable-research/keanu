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

    public static ForwardModePartialDerivative correctForBroadcastPartialForward(ForwardModePartialDerivative partial, long[] partialOfShape, long[] targetOfShape) {

        if (shouldCorrectPartialForBroadcast(partial, partialOfShape, targetOfShape)) {
            return broadcastPartialForward(partial, partialOfShape, targetOfShape);
        } else {
            return partial;
        }
    }

    public static ForwardModePartialDerivative broadcastPartialForward(ForwardModePartialDerivative partial, long[] partialOfShape, long[] targetOfShape) {
        long[] wrtShape = partial.getWrtShape();
        long[] partialReshape = TensorShape.concat(wrtShape, TensorShape.shapeToDesiredRankByPrependingOnes(partialOfShape, targetOfShape.length));
        long[] resultShape = TensorShape.concat(wrtShape, targetOfShape);

        DoubleTensor correctedPartial = partial.get().reshape(partialReshape).broadcast(resultShape);

        return new ForwardModePartialDerivative(wrtShape, correctedPartial);
    }

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

    /**
     * @param partial       The partial derivative that may or may not come from a broadcasted operation.
     * @param actualShape   The part of the partial shape that should match the expected shape in the case no broadcast was
     *                      performed. This would be the of shape for forward mode and the with respect to shape for reverse.
     * @param expectedShape The shape of the operation result in forward mode or the shape of the operand in reverse mode.
     *                      This should match the actual shape if no broadcast was performed.
     * @return true if a broadcast should be taken into account and corrected for in the auto diff calculation, false otherwise.
     */
    private static boolean shouldCorrectPartialForBroadcast(ForwardModePartialDerivative partial, long[] actualShape, long[] expectedShape) {
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

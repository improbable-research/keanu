package io.improbable.keanu.tensor;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.List;

import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getShapeIndices;

/**
 * This class provides shape calculation and index mapping for the slice operation.
 * Given a description of slices (Slicer) and a source shape and stride, the shape of
 * the slice operation can be calculated.
 * <p>
 * Ths class stores the result shape without dropping the dimensions that are ultimately dropped
 * in order to more efficiently calculate source to result index mappings.
 * <p>
 * Dimensions are dropped for example given 1:2,3 then the 2nd dimension length will be 1 and ALSO dropped but
 * the first dimension will be kept despite also being length one. This is done this way to mimic numpy behaviour.
 */
public final class SliceIndexMapper {

    private final Slicer slicer;


    private final long[] sourceShape;
    private final long[] sourceStride;

    private final long[] resultShapeWithoutRankLoss;
    private final long[] resultStrideWithoutRankLoss;
    private final int[] dimensionsDropped;

    public SliceIndexMapper(Slicer slicer, long[] sourceShape, long[] sourceStride) {
        this.slicer = slicer;
        this.sourceShape = sourceShape;
        this.sourceStride = sourceStride;

        List<Long> shapeList = new ArrayList<>();
        List<Integer> droppedList = new ArrayList<>();
        List<Slicer.StartStopStep> slices = slicer.getSlices();
        initialize(sourceShape, shapeList, droppedList, slices);

        this.resultShapeWithoutRankLoss = Longs.toArray(shapeList);
        this.resultStrideWithoutRankLoss = TensorShape.getRowFirstStride(resultShapeWithoutRankLoss);
        this.dimensionsDropped = Ints.toArray(droppedList);
    }

    private static void initialize(long[] sourceShape, List<Long> shapeList, List<Integer> droppedList, List<Slicer.StartStopStep> slices) {
        Preconditions.checkArgument(slices.size() <= sourceShape.length, "Too many slices specified for shape");

        for (int i = 0; i < sourceShape.length; i++) {

            Slicer.StartStopStep startStopStep = slices.get(i);

            if (startStopStep.getStop() != Slicer.StartStopStep.START_PLUS_ONE_STOP) {
                if (startStopStep == Slicer.StartStopStep.ALL) {
                    shapeList.add(sourceShape[i]);
                } else {
                    final long absStep = Math.abs(startStopStep.getStep());
                    final long minStop = Math.min(startStopStep.getStop(), sourceShape[i]);
                    final long length = 1 + (minStop - 1 - startStopStep.getStart()) / absStep;
                    shapeList.add(length);
                }
            } else {
                shapeList.add(1L);
                droppedList.add(i);
            }
        }
    }

    /**
     * @return The resultShapeWithoutRankLoss with the dropped dimensions dropped.
     */
    public long[] getResultShape() {
        return dimensionsDropped.length > 0 ? ArrayUtils.removeAll(resultShapeWithoutRankLoss, dimensionsDropped) : resultShapeWithoutRankLoss;
    }

    /**
     * @return The resultStrideWithoutRankLoss with the dropped dimensions dropped.
     */
    public long[] getResultStride() {
        return dimensionsDropped.length > 0 ? ArrayUtils.removeAll(resultStrideWithoutRankLoss, dimensionsDropped) : resultStrideWithoutRankLoss;
    }

    /**
     * @param resultIndex the index in the result buffer
     * @return the index in the source buffer that maps to the result buffer.
     */
    public long getSourceIndexFromResultIndex(long resultIndex) {

        final long[] shapeIndices = getShapeIndices(resultShapeWithoutRankLoss, resultStrideWithoutRankLoss, resultIndex);
        final long[] sourceIndices = getIndicesOfSource(shapeIndices);

        return getFlatIndex(sourceShape, sourceStride, sourceIndices);
    }

    /**
     * warning: This method mutates the argument.
     *
     * @param shapeIndices indices from result array
     * @return the indices of the source that maps to the shapeIndices
     */
    private long[] getIndicesOfSource(final long[] shapeIndices) {

        final List<Slicer.StartStopStep> slices = slicer.getSlices();
        for (int i = 0; i < slices.size(); i++) {
            if (resultShapeWithoutRankLoss[i] != sourceShape[i]) {
                final Slicer.StartStopStep slice = slices.get(i);

                shapeIndices[i] = slice.getStart() + shapeIndices[i] * slice.getStep();
            }
        }

        return shapeIndices;
    }
}

package io.improbable.keanu.tensor.jvm;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import lombok.Value;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.List;

@Value
public class Slicer {

    public Slicer(List<Slice> slices, Integer ellipsisPosition) {
        this.slices = slices;
        this.ellipsisPosition = ellipsisPosition;
    }

    public Slicer(long[] start, long[] end, long[] stride, Integer ellipsis, boolean[] upperBoundStop, boolean[] dropDimension) {
        this.slices = new ArrayList<>();
        this.ellipsisPosition = ellipsis;

        for (int i = 0; i < start.length; i++) {
            Slice slice = new Slice(start[i], upperBoundStop[i] ? null : end[i], stride[i], dropDimension[i]);
            this.slices.add(slice);
        }
    }

    public static SlicerBuilder builder() {
        return new SlicerBuilder();
    }

    @Value
    public static class Slice {
        public static final long DEFAULT_START = 0;
        public static final long DEFAULT_STEP = 1;
        public static final Slice ALL = new Slice(DEFAULT_START, null, DEFAULT_STEP, false);

        private boolean upperBoundStop;
        private boolean dropDimension;
        private Long start;
        private Long stop;
        private Long step;

        Slice(Number start, Number stop, Number step, boolean dropDimension) {
            this.start = start != null ? start.longValue() : DEFAULT_START;
            this.stop = stop != null ? stop.longValue() : null;
            this.step = step != null ? step.longValue() : DEFAULT_STEP;
            Preconditions.checkArgument(this.step != 0, "Step cannot be 0");
            this.upperBoundStop = stop == null;
            this.dropDimension = dropDimension;
        }

        public Long getStop(long dimLength) {
            if (upperBoundStop) {
                return dimLength;
            } else if (stop < 0) {
                return Math.max(0, dimLength + stop);
            } else {
                return Math.min(dimLength, stop);
            }
        }

        public Long getStart(long dimLength) {
            if (start < 0) {
                return Math.max(0, dimLength + start);
            } else {
                return start;
            }
        }
    }

    private final List<Slice> slices;
    private final Integer ellipsisPosition;

    /**
     * @param dimension        the dimension for which the slice is requested
     * @param givenShapeLength the rank of the tensor being sliced. This is needed due to the ellipsis making this relative to a shape.
     * @return a slice for a given dimension
     */
    public Slice getSlice(int dimension, int givenShapeLength) {

        final int ellipsisStart;
        final int ellipsisEnd;
        final int ellipsisDimCount;

        if (ellipsisPosition != null) {
            ellipsisStart = ellipsisPosition;
            ellipsisDimCount = givenShapeLength - slices.size();
            ellipsisEnd = ellipsisStart + ellipsisDimCount;
        } else {
            ellipsisStart = 0;
            ellipsisDimCount = 0;
            ellipsisEnd = 0;
        }

        final boolean takeAllByEllipsis = dimension >= ellipsisStart && dimension < ellipsisEnd;

        if (takeAllByEllipsis) {
            return Slice.ALL;
        } else {
            final int indexWithEllipsis = ellipsisPosition != null && dimension >= ellipsisEnd ? dimension - ellipsisDimCount : dimension;

            if (indexWithEllipsis >= slices.size()) {
                return Slice.ALL;
            } else {
                return slices.get(indexWithEllipsis);
            }
        }
    }

    /**
     * @param givenShape the shape being sliced
     * @param keepDims   true if no rank change is wanted false otherwise
     * @return the resulting shape when this slicer is applied to the given shape
     */
    public long[] getResultShape(final long[] givenShape, final boolean keepDims) {

        final long[] resultShapeWithoutRankLoss = new long[givenShape.length];

        for (int i = 0; i < givenShape.length; i++) {

            Slice slice = getSlice(i, givenShape.length);

            if (slice == Slice.ALL) {
                resultShapeWithoutRankLoss[i] = givenShape[i];
            } else {

                if (slice.isDropDimension()) {
                    resultShapeWithoutRankLoss[i] = 1L;

                } else {
                    resultShapeWithoutRankLoss[i] = getSliceLength(givenShape[i], slice);
                }
            }

        }

        if (keepDims) {
            return resultShapeWithoutRankLoss;
        } else {
            return ArrayUtils.removeAll(resultShapeWithoutRankLoss, getDroppedDimensions(givenShape));
        }
    }

    /**
     * @param dimLength the given dimension length for which this slice will be applied
     * @param slice     the slice object that describes how the dimension will be sliced and therefore determines the
     *                  length of the resulting dimension.
     * @return the length of the dimension after the slice is applied
     */
    private long getSliceLength(long dimLength, Slice slice) {

        final long stop = slice.getStop(dimLength);
        final long start = slice.getStart(dimLength);
        final long excludeStopOffset = (long) Math.signum(slice.getStep());
        final long stepCount = (stop - excludeStopOffset - start) / slice.getStep();

        if (stepCount >= 0) {
            return 1 + stepCount;
        } else {
            return 0;
        }
    }

    /**
     * @param givenShape the shape that the slice is being applied to
     * @return the dimensions that will be dropped when the slicer is applied
     */
    public int[] getDroppedDimensions(long[] givenShape) {

        List<Integer> droppedDimensions = new ArrayList<>();

        for (int i = 0; i < givenShape.length; i++) {
            if (getSlice(i, givenShape.length).isDropDimension()) {
                droppedDimensions.add(i);
            }
        }

        return Ints.toArray(droppedDimensions);
    }

    public static class SlicerBuilder {
        private ArrayList<Slice> slices = new ArrayList<>();
        private Integer ellipsisPosition = null;

        /**
         * @param start index to start at
         * @param stop  index to stop at
         * @param step  step from start to stop
         * @return a slice for dimension that takes all indices after (inclusively) the start and up
         * to the stop (exclusively) with a step of step.
         */
        public SlicerBuilder slice(Long start, Long stop, Long step) {
            this.slices.add(new Slice(start, stop, step, false));
            return this;
        }

        public SlicerBuilder slice(Integer start, Integer stop, Integer step) {
            this.slices.add(new Slice(start, stop, step, false));
            return this;
        }

        /**
         * @param start index to start at
         * @param stop  index to stop at
         * @return a slice for dimension that takes all indices after (inclusively) the start and up
         * to the stop (exclusively).
         */
        public SlicerBuilder slice(Long start, Long stop) {
            this.slices.add(new Slice(start, stop, Slice.DEFAULT_STEP, false));
            return this;
        }

        public SlicerBuilder slice(Integer start, Integer stop) {
            this.slices.add(new Slice(start, stop, (int) Slice.DEFAULT_STEP, false));
            return this;
        }

        /**
         * @param start index to start at
         * @return a slice for dimension that takes all indices after (inclusively) the start.
         */
        public SlicerBuilder slice(Long start) {
            this.slices.add(new Slice(start, null, null, true));
            return this;
        }

        public SlicerBuilder slice(Integer start) {
            this.slices.add(new Slice(start, null, null, true));
            return this;
        }

        /**
         * Take all indices. This is the same as all().
         *
         * @return a slice for dimension that takes all indices
         */
        public SlicerBuilder slice() {
            this.slices.add(Slice.ALL);
            return this;
        }

        /**
         * Take all indices. This is the same as slice().
         *
         * @return a slice for dimension that takes all indices
         */
        public SlicerBuilder all() {
            this.slices.add(Slice.ALL);
            return this;
        }

        /**
         * Means take all from this point. This can only be used once and could for example allow
         * slicing dimensions from the right. e.g. a given shape of (2,3,4) could be sliced [...,2]
         * which would be the same as [:,:,2].
         *
         * @return a slice for dimension that takes all indices
         */
        public SlicerBuilder ellipsis() {
            if (ellipsisPosition != null) {
                throw new IllegalStateException("Only 1 ellipsis is allowed.");
            }
            ellipsisPosition = slices.size();
            return this;
        }

        public Slicer build() {
            return new Slicer(slices, ellipsisPosition);
        }
    }

    public static Slicer fromString(String sliceArg) {

        String cleanSliceArg = sliceArg.replaceAll("\\s+", "");
        String[] byDimension = cleanSliceArg.split(",");

        SlicerBuilder slicerBuilder = Slicer.builder();

        for (String dimSlice : byDimension) {

            if (dimSlice.equals("...")) {
                slicerBuilder.ellipsis();
            } else {
                String[] byColon = dimSlice.split(":", 3);

                if (byColon.length == 3) {
                    slicerBuilder.slice(parseStart(byColon[0]), parseStop(byColon[1]), parseStep(byColon[2]));
                } else if (byColon.length == 2) {
                    slicerBuilder.slice(parseStart(byColon[0]), parseStop(byColon[1]));
                } else if (byColon.length == 1) {
                    slicerBuilder.slice(parseStart(byColon[0]));
                } else if (byColon.length == 0) {
                    slicerBuilder.all();
                } else {
                    throw new IllegalArgumentException("Can not parse slice " + dimSlice);
                }
            }
        }

        return slicerBuilder.build();
    }

    private static Long parseStart(String arg) {
        return getOrDefault(arg, Slice.DEFAULT_START);
    }

    private static Long parseStop(String arg) {
        return getOrDefault(arg, null);
    }

    private static Long parseStep(String arg) {
        return getOrDefault(arg, Slice.DEFAULT_STEP);
    }

    private static Long getOrDefault(String arg, Long defaultValue) {
        try {
            return Long.parseLong(arg);
        } catch (NumberFormatException nfe) {
            return defaultValue;
        }
    }
}

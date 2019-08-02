package io.improbable.keanu.tensor.jvm;

import lombok.Value;

import java.util.ArrayList;
import java.util.List;

@Value
public class Slicer {

    private Slicer(List<StartStopStep> slices) {
        this.slices = slices;
    }

    public static SlicerBuilder builder() {
        return new SlicerBuilder();
    }

    @Value
    public static class StartStopStep {
        public static final long DEFAULT_START = 0;
        public static final long UPPER_BOUND_STOP = -1;
        public static final long START_PLUS_ONE_STOP = -2;
        public static final long DEFAULT_STEP = 1;
        public static final StartStopStep ALL = new StartStopStep(DEFAULT_START, UPPER_BOUND_STOP, DEFAULT_STEP);

        long start;
        long stop;
        long step;

        StartStopStep(Long start, Long stop, Long step) {
            this.start = start != null ? start : DEFAULT_START;
            this.stop = stop != null ? stop : UPPER_BOUND_STOP;
            this.step = step != null ? step : DEFAULT_STEP;
        }

        StartStopStep(Integer start, Integer stop, Integer step) {
            this.start = start != null ? start : DEFAULT_START;
            this.stop = stop != null ? stop : UPPER_BOUND_STOP;
            this.step = step != null ? step : DEFAULT_STEP;
        }
    }

    private final List<StartStopStep> slices;

    public static class SlicerBuilder {
        private ArrayList<StartStopStep> slices = new ArrayList<>();

        /**
         * @param start index to start at
         * @param stop  index to stop at
         * @param step  step from start to stop
         * @return a slice for dimension that takes all indices after (inclusively) the start and up
         * to the stop (exclusively) with a step of step.
         */
        public SlicerBuilder slice(Long start, Long stop, Long step) {
            this.slices.add(new StartStopStep(start, stop, step));
            return this;
        }

        public SlicerBuilder slice(Integer start, Integer stop, Integer step) {
            this.slices.add(new StartStopStep(start, stop, step));
            return this;
        }

        /**
         * @param start index to start at
         * @param stop  index to stop at
         * @return a slice for dimension that takes all indices after (inclusively) the start and up
         * to the stop (exclusively).
         */
        public SlicerBuilder slice(Long start, Long stop) {
            this.slices.add(new StartStopStep(start, stop, StartStopStep.DEFAULT_STEP));
            return this;
        }

        public SlicerBuilder slice(Integer start, Integer stop) {
            this.slices.add(new StartStopStep(start, stop, (int) StartStopStep.DEFAULT_STEP));
            return this;
        }

        /**
         * @param start index to start at
         * @return a slice for dimension that takes all indices after (inclusively) the start.
         */
        public SlicerBuilder slice(Long start) {
            this.slices.add(new StartStopStep(start, StartStopStep.START_PLUS_ONE_STOP, StartStopStep.DEFAULT_STEP));
            return this;
        }

        public SlicerBuilder slice(Integer start) {
            this.slices.add(new StartStopStep(start, (int) StartStopStep.START_PLUS_ONE_STOP, (int) StartStopStep.DEFAULT_STEP));
            return this;
        }

        /**
         * Take all indices. This is the same as all().
         *
         * @return a slice for dimension that takes all indices
         */
        public SlicerBuilder slice() {
            this.slices.add(StartStopStep.ALL);
            return this;
        }

        /**
         * Take all indices. This is the same as slice().
         *
         * @return a slice for dimension that takes all indices
         */
        public SlicerBuilder all() {
            this.slices.add(StartStopStep.ALL);
            return this;
        }

        public Slicer build() {
            return new Slicer(slices);
        }
    }

    public static Slicer fromString(String sliceArg) {

        String cleanSliceArg = sliceArg.replaceAll("\\s+", "");
        String[] byDimension = cleanSliceArg.split(",");

        SlicerBuilder slicerBuilder = Slicer.builder();

        for (String dimSlice : byDimension) {

            if (dimSlice.equals("...")) {
                slicerBuilder.all();
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
        return getOrDefault(arg, StartStopStep.DEFAULT_START);
    }

    private static Long parseStop(String arg) {
        return getOrDefault(arg, StartStopStep.UPPER_BOUND_STOP);
    }

    private static Long parseStep(String arg) {
        return getOrDefault(arg, StartStopStep.DEFAULT_STEP);
    }

    private static Long getOrDefault(String arg, long defaultValue) {
        try {
            return Long.parseLong(arg);
        } catch (NumberFormatException nfe) {
            return defaultValue;
        }
    }
}

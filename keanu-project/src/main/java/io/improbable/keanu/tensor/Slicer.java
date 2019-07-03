package io.improbable.keanu.tensor;

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
        public static final long DEFAULT_STOP = -1;
        public static final long DEFAULT_STEP = 1;
        public static final StartStopStep ALL = new StartStopStep(DEFAULT_START, DEFAULT_STOP, DEFAULT_STEP);

        long start;
        long stop;
        long step;

        public StartStopStep(Long start, Long stop, Long step) {
            this.start = start != null ? start : DEFAULT_START;
            this.stop = stop != null ? stop : DEFAULT_STOP;
            this.step = step != null ? step : DEFAULT_STEP;
        }

        public StartStopStep(Integer start, Integer stop, Integer step) {
            this.start = start != null ? start : DEFAULT_START;
            this.stop = stop != null ? stop : DEFAULT_STOP;
            this.step = step != null ? step : DEFAULT_STEP;
        }
    }

    private final List<StartStopStep> slices;

    public static class SlicerBuilder {
        private ArrayList<StartStopStep> slices = new ArrayList<>();

        /**
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
         * @return a slice for dimension that takes all indices after (inclusively) the start.
         */
        public SlicerBuilder slice(Long start) {
            this.slices.add(new StartStopStep(start, StartStopStep.DEFAULT_STOP, StartStopStep.DEFAULT_STEP));
            return this;
        }

        public SlicerBuilder slice(Integer start) {
            this.slices.add(new StartStopStep(start, (int) StartStopStep.DEFAULT_STOP, (int) StartStopStep.DEFAULT_STEP));
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
                String[] byColon = dimSlice.split(":");

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
        return getOrDefault(arg, StartStopStep.DEFAULT_STOP);
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

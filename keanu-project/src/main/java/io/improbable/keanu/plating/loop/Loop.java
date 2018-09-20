package io.improbable.keanu.plating.loop;

import java.util.Collection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.plating.Plate;
import io.improbable.keanu.plating.PlateBuilder;
import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;

/**
 * A Loop object is a convenient wrapper around some Plates.
 * See LoopTest.java for examples of how it's used.
 * The way it works is to unroll the loop to a given maximum size and evaluate it in full
 * (so it's not very performant)
 * <p>
 * The resulting graph structure looks like this.
 * "base" is the base case, provided by the user
 * "cond" is the condition (one instance per iteration), provided by the user
 * "iterate" is the iteration step, provided by the user
 * "V" is the output of the loop
 * "L" indicates if it's still looping. This is used to detect the error case in which the loop was too short to complete.
 * <p>
 *       cond true base
 *           \ |   |  \
 *            AND  | iterate
 *             |\  |  /|
 *             | \ | / |
 *       cond  |  IF   |
 *           \ |   |\  |
 *            AND  | iterate
 *             |\  |  /|
 *             | \ | / |
 *       cond  |  IF   |
 *           \ |   |\  |
 *            AND  | iterate
 *             |\  |  /
 *             | \ | /
 *             |  IF
 *             |   |
 *             L   V
 */
public class Loop {
    private Logger log = LoggerFactory.getLogger(this.getClass());
    public static final VertexLabel VALUE_OUT_LABEL = new VertexLabel("loop_value_out");
    public static final VertexLabel VALUE_IN_LABEL = PlateBuilder.proxyFor(VALUE_OUT_LABEL);
    public static final VertexLabel CONDITION_LABEL = new VertexLabel("loop_condition");
    public static final VertexLabel LOOP_LABEL = new VertexLabel("loop");
    static final VertexLabel STILL_LOOPING_LABEL = PlateBuilder.proxyFor(LOOP_LABEL);
    public static final int DEFAULT_MAX_COUNT = 100;
    private final Plates plates;
    private final boolean throwWhenMaxCountIsReached;

    /**
     * package-private because it is intended to be created by the LoopBuilder
     *
     * @param plates the set of plates, one for each iteration in the loop
     * @param throwWhenMaxCountIsReached optionally disable throwing and log a warning instead
     */
    Loop(Plates plates, boolean throwWhenMaxCountIsReached) {
        this.plates = plates;
        this.throwWhenMaxCountIsReached = throwWhenMaxCountIsReached;
    }

    public Plates getPlates() {
        return plates;
    }

    /**
     * A factory method for creating a loop
     * It automatically labels the initial state correctly for you.
     *
     * @param initialState a single Vertex that defines the loop's base case.
     * @param <V>          the input type
     * @return a builder object
     */
    public static <V extends Vertex<?>> LoopBuilder startingFrom(V initialState) {
        if (initialState.getLabel() == null) {
            initialState.setLabel(VALUE_OUT_LABEL);
        }
        return startingFrom(ImmutableList.of(initialState));
    }

    /**
     * A factory method for creating a loop
     *
     * @param first  the first Vertex (mandatory)
     * @param others other Vertices (optional)
     * @param <V>    the input type
     * @return a builder object
     */
    public static <V extends Vertex<?>> LoopBuilder startingFrom(V first, V... others) {
        return startingFrom(ImmutableList.<V>builder().add(first).add(others).build());
    }

    /**
     * A factory method for creating a loop
     *
     * @param initialState the collection of vertices that define the loop's base case
     * @param <V>          the input type
     * @return a builder object
     */
    public static <V extends Vertex<?>> LoopBuilder startingFrom(Collection<V> initialState) {
        return new LoopBuilder(initialState);
    }

    /**
     * @param <V> the output type
     * @return the output of the Loop (i.e. the output Vertex from the final Plate)
     * @throws LoopException if the loop was too short and hit its maximum unrolled size
     */
    public <V extends Vertex<?>> V getOutput() throws LoopException {
        Plate finalPlate = plates.getLastPlate();
        checkIfMaxCountHasBeenReached(finalPlate);
        return finalPlate.get(VALUE_OUT_LABEL);
    }

    private void checkIfMaxCountHasBeenReached(Plate plate) throws LoopException {
        BoolVertex stillLooping = plate.get(STILL_LOOPING_LABEL);
        if (!stillLooping.getValue().allFalse()) {
            String errorMessage = "Loop has exceeded its max count " + plates.size();
            if (throwWhenMaxCountIsReached) {
                throw new LoopException(errorMessage);
            } else {
                log.info(errorMessage);
            }
        }
    }
}

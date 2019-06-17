package io.improbable.keanu.templating.loop;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.templating.Sequence;
import io.improbable.keanu.templating.SequenceBuilder;
import io.improbable.keanu.templating.SequenceItem;
import io.improbable.keanu.vertices.SimpleVertexDictionary;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import lombok.extern.slf4j.Slf4j;

import java.util.Map;

/**
 * A Loop object is a convenient wrapper around some Sequence.
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
 * cond true base
 * \ |   |  \
 * AND  | iterate
 * |\  |  /|
 * | \ | / |
 * cond  |  IF   |
 * \ |   |\  |
 * AND  | iterate
 * |\  |  /|
 * | \ | / |
 * cond  |  IF   |
 * \ |   |\  |
 * AND  | iterate
 * |\  |  /
 * | \ | /
 * |  IF
 * |   |
 * L   V
 */
@Slf4j
public class Loop {
    public static final VertexLabel VALUE_OUT_LABEL = new VertexLabel("loop_value_out");
    public static final VertexLabel CONDITION_LABEL = new VertexLabel("loop_condition");
    public static final VertexLabel VALUE_IN_LABEL = SequenceBuilder.proxyLabelFor(VALUE_OUT_LABEL);
    public static final VertexLabel STILL_LOOPING_LABEL = SequenceBuilder.proxyLabelFor(new VertexLabel("loop"));
    public static final int DEFAULT_MAX_COUNT = 100;
    private final Sequence sequence;
    private final boolean throwWhenMaxCountIsReached;

    /**
     * package-private because it is intended to be created by the LoopBuilder
     *
     * @param sequence                   the set of sequence, one for each iteration in the loop
     * @param throwWhenMaxCountIsReached optionally disable throwing and log a warning instead
     */
    Loop(Sequence sequence, boolean throwWhenMaxCountIsReached) {
        this.sequence = sequence;
        this.throwWhenMaxCountIsReached = throwWhenMaxCountIsReached;
    }

    public Sequence getSequence() {
        return sequence;
    }

    /**
     * A factory method for creating a loop
     *
     * @param first  the first Vertex (mandatory)
     * @param others other Vertices (optional)
     * @param <V>    the input type
     * @return a builder object
     */
    public static <V extends Vertex<?>> LoopBuilder withInitialConditions(V first, V... others) {
        Map<VertexLabel, Vertex<?>> map = buildMapForBaseCase(first, others);
        return withInitialConditions(SimpleVertexDictionary.backedBy(map));
    }

    /**
     * A factory method for creating a loop
     *
     * @param initialState the collection of vertices that define the loop's base case
     * @return a builder object
     */
    public static LoopBuilder withInitialConditions(VertexDictionary initialState) {
        return new LoopBuilder(initialState);
    }

    private static <V extends Vertex<?>> Map<VertexLabel, Vertex<?>> buildMapForBaseCase(V first, V[] others) {
        ImmutableMap.Builder<VertexLabel, Vertex<?>> baseCaseMap = ImmutableMap.builder();
        baseCaseMap.put(VALUE_OUT_LABEL, first);
        for (V vertex : others) {
            VertexLabel label = vertex.getLabel();
            if (label == null) {
                label = new VertexLabel(String.format("base_case_vertex_%d", vertex.hashCode()));
            }
            baseCaseMap.put(label, vertex);
        }
        try {
            return baseCaseMap.build();
        } catch (IllegalArgumentException e) {
            throw new LoopConstructionException("Duplicate label found in base case");
        }
    }

    /**
     * @param <V> the output type
     * @return the output of the Loop (i.e. the output Vertex from the final SequenceItem)
     * @throws LoopDidNotTerminateException if the loop was too short and hit its maximum unrolled size
     */
    public <V extends Vertex<?>> V getOutput() throws LoopDidNotTerminateException {
        SequenceItem finalItem = sequence.getLastItem();
        checkIfMaxCountHasBeenReached(finalItem);
        return finalItem.get(VALUE_OUT_LABEL);
    }

    private void checkIfMaxCountHasBeenReached(SequenceItem item) throws LoopDidNotTerminateException {
        BooleanVertex stillLooping = item.get(STILL_LOOPING_LABEL);
        if (!stillLooping.getValue().allFalse()) {
            String errorMessage = "Loop has exceeded its max count " + sequence.size();
            if (throwWhenMaxCountIsReached) {
                throw new LoopDidNotTerminateException(errorMessage);
            } else {
                log.info(errorMessage);
            }
        }
    }
}

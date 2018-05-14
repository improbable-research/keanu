package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;

public class BoolReduceVertex extends NonProbabilisticBool {
    private final List<? extends Vertex<Boolean>> inputs;
    private final BiFunction<Boolean, Boolean, Boolean> reduceFunction;

    public BoolReduceVertex(Collection<Vertex<Boolean>> input, BiFunction<Boolean, Boolean, Boolean> reduceFunction) {
        if (input.size() < 2) {
            throw new IllegalArgumentException("BoolReduceVertex should have at least two input vertices, called with " + input.size());
        }

        this.inputs = new ArrayList<>(input);
        this.reduceFunction = reduceFunction;
        setParents(inputs);
    }

    public BoolReduceVertex(BiFunction<Boolean, Boolean, Boolean> f, Vertex<Boolean>... input) {
        this(Arrays.asList(input), f);
    }

    @Override
    public Boolean sample(Random random) {
        return applyReduce((vertex) -> vertex.sample(random));
    }

    @Override
    public Boolean lazyEval() {
        setValue(applyReduce(Vertex::lazyEval));
        return getValue();
    }

    @Override
    public Boolean getDerivedValue() {
        return applyReduce(Vertex::getValue);
    }

    private boolean applyReduce(Function<Vertex<Boolean>, Boolean> mapper) {
        Iterator<? extends Vertex<Boolean>> vertices = inputs.iterator();

        boolean c = mapper.apply(vertices.next());
        while (vertices.hasNext()) {
            c = reduceFunction.apply(c, mapper.apply(vertices.next()));
        }
        return c;
    }
}

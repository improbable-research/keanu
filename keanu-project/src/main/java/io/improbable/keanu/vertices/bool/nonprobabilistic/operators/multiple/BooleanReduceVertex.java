package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public class BooleanReduceVertex extends BooleanVertex implements NonProbabilistic<BooleanTensor>, NonSaveableVertex {
    private final List<? extends Vertex<BooleanTensor>> inputs;
    private final BiFunction<BooleanTensor, BooleanTensor, BooleanTensor> reduceFunction;

    public BooleanReduceVertex(long[] shape, Collection<? extends Vertex<BooleanTensor>> input,
                               BiFunction<BooleanTensor, BooleanTensor, BooleanTensor> reduceFunction) {
        super(shape);
        if (input.size() < 2) {
            throw new IllegalArgumentException("BooleanReduceVertex should have at least two input vertices, called with " + input.size());
        }

        this.inputs = new ArrayList<>(input);
        this.reduceFunction = reduceFunction;
        setParents(inputs);
    }


    public BooleanReduceVertex(long[] shape, BiFunction<BooleanTensor, BooleanTensor, BooleanTensor> f, Vertex<BooleanTensor>... input) {
        this(shape, Arrays.asList(input), f);
    }

    @Override
    public BooleanTensor calculate() {
        return applyReduce(Vertex::getValue);
    }

    private BooleanTensor applyReduce(Function<Vertex<BooleanTensor>, BooleanTensor> mapper) {
        Iterator<? extends Vertex<BooleanTensor>> vertices = inputs.iterator();

        BooleanTensor c = mapper.apply(vertices.next());
        while (vertices.hasNext()) {
            c = reduceFunction.apply(c, mapper.apply(vertices.next()));
        }
        return c;
    }
}

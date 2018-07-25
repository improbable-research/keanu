package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class BooleanReduceVertex extends BooleanVertex {
    private final List<? extends Vertex<BooleanTensor>> inputs;
    private final BiFunction<BooleanTensor, BooleanTensor, BooleanTensor> reduceFunction;

    public BooleanReduceVertex(int[] shape, Collection<Vertex<BooleanTensor>> input,
                               BiFunction<BooleanTensor, BooleanTensor, BooleanTensor> reduceFunction) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((BooleanReduceVertex) v).applyReduce(Vertex::getValue)),
            Observable.observableTypeFor(BooleanReduceVertex.class)
        );
        if (input.size() < 2) {
            throw new IllegalArgumentException("BooleanReduceVertex should have at least two input vertices, called with " + input.size());
        }

        this.inputs = new ArrayList<>(input);
        this.reduceFunction = reduceFunction;
        setParents(inputs);
        setValue(BooleanTensor.placeHolder(shape));
    }


    public BooleanReduceVertex(int[] shape, BiFunction<BooleanTensor, BooleanTensor, BooleanTensor> f, Vertex<BooleanTensor>... input) {
        this(shape, Arrays.asList(input), f);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return applyReduce((vertex) -> vertex.sample(random));
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

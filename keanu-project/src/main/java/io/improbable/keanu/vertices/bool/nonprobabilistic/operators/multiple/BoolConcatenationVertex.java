package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.function.Function;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class BoolConcatenationVertex extends BooleanVertex {

    private final int dimension;
    private final BooleanVertex[] input;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different.
     * @param input the input vertices to concatenate
     */
    public BoolConcatenationVertex(int dimension, BooleanVertex... input) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((BoolConcatenationVertex) v).applyConcat(Vertex::getValue)),
            Observable.observableTypeFor(BoolConcatenationVertex.class)
        );
        this.dimension = dimension;
        this.input = input;
        setParents(input);
        int[][] shapes = extractFromInputs(int[].class, Vertex::getShape);
        setValue(BooleanTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    private BooleanTensor applyConcat(Function<Vertex<BooleanTensor>, BooleanTensor> mapper) {
        BooleanTensor[] inputs = extractFromInputs(BooleanTensor.class, mapper);
        BooleanTensor primary = inputs[0];
        BooleanTensor[] toConcat = Arrays.copyOfRange(inputs, 1, inputs.length);
        return primary.concat(dimension, toConcat);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return applyConcat(Vertex::sample);
    }

    private <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<BooleanTensor>, T> func) {
        T[] extract = (T[]) Array.newInstance(clazz, input.length);
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(input[i]);
        }
        return extract;
    }

}

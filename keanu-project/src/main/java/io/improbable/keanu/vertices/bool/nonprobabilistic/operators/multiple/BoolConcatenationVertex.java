package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

public class BoolConcatenationVertex extends NonProbabilisticBool {

    private final int dimension;
    private final BoolVertex[] input;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different.
     * @param input the input vertices to concatenate
     */
    public BoolConcatenationVertex(int dimension, BoolVertex... input) {
        this.dimension = dimension;
        this.input = input;
        setParents(input);
        int[][] shapes = extractFromInputs(int[].class, Vertex::getShape);
        setValue(BooleanTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    @Override
    public BooleanTensor getDerivedValue() {
        return op(extractFromInputs(BooleanTensor.class, Vertex::getValue));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(extractFromInputs(BooleanTensor.class, Vertex::sample));
    }

    protected BooleanTensor op(BooleanTensor... inputs) {
        BooleanTensor primary = inputs[0];
        BooleanTensor[] toConcat = Arrays.copyOfRange(inputs, 1, inputs.length);
        return primary.concat(dimension, toConcat);
    }

    private <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<BooleanTensor>, T> func) {
        T[] extract = (T[]) Array.newInstance(clazz, input.length);
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(input[i]);
        }
        return extract;
    }

}

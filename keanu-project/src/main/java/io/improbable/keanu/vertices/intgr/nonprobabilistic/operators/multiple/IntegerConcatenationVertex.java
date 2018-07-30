package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.number.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.NonProbabilisticInteger;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

public class IntegerConcatenationVertex extends NonProbabilisticInteger {

    private final int dimension;
    private final IntegerVertex[] input;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different.
     * @param input the input vertices to concatenate
     */
    public IntegerConcatenationVertex(int dimension, IntegerVertex... input) {
        this.dimension = dimension;
        this.input = input;
        setParents(input);
        int[][] shapes = extractFromInputs(int[].class, Vertex::getShape);
        setValue(IntegerTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    @Override
    public IntegerTensor getDerivedValue() {
        return op(extractFromInputs(IntegerTensor.class, Vertex::getValue));
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op(extractFromInputs(IntegerTensor.class, Vertex::sample));
    }

    protected IntegerTensor op(IntegerTensor... inputs) {
        IntegerTensor primary = inputs[0];
        IntegerTensor[] toConcat = Arrays.copyOfRange(inputs, 1, inputs.length);
        return primary.concat(dimension, toConcat);
    }

    private <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<IntegerTensor>, T> func) {
        T[] extract = (T[]) Array.newInstance(clazz, input.length);
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(input[i]);
        }
        return extract;
    }

}

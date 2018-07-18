package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.NonProbabilisticInteger;

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
        int[][] shapes = new int[input.length][];
        for (int i = 0; i < input.length; i++) shapes[i] = input[i].getShape();
        setValue(IntegerTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    @Override
    public IntegerTensor getDerivedValue() {
        return op(extractFromInputs(i -> input[i].getValue()));
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op(extractFromInputs(i -> input[i].sample()));
    }

    protected IntegerTensor op(IntegerTensor... inputs) {
        IntegerTensor primary = inputs[0];
        IntegerTensor[] toConcat = Arrays.copyOfRange(inputs, 1, inputs.length);
        return primary.concat(dimension, toConcat);
    }

    private IntegerTensor[] extractFromInputs(Function<Integer, IntegerTensor> func) {
        IntegerTensor[] extract = new IntegerTensor[input.length];
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(i);
        }
        return extract;
    }

}

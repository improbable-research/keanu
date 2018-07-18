package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

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
        int[][] shapes = new int[input.length][];
        for (int i = 0; i < input.length; i++) shapes[i] = input[i].getShape();
        setValue(BooleanTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    @Override
    public BooleanTensor getDerivedValue() {
        return op(extractFromInputs(i -> input[i].getValue()));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(extractFromInputs(i -> input[i].sample()));
    }

    protected BooleanTensor op(BooleanTensor... inputs) {
        BooleanTensor primary = inputs[0];
        BooleanTensor[] toConcat = Arrays.copyOfRange(inputs, 1, inputs.length);
        return primary.concat(dimension, toConcat);
    }

    private BooleanTensor[] extractFromInputs(Function<Integer, BooleanTensor> func) {
        BooleanTensor[] extract = new BooleanTensor[input.length];
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(i);
        }
        return extract;
    }

}

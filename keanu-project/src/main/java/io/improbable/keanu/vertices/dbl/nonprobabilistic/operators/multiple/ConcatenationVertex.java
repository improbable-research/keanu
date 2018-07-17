package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

public class ConcatenationVertex extends NonProbabilisticDouble {

    private final int dimension;
    private final DoubleVertex[] input;

    public ConcatenationVertex(int dimension, DoubleVertex... input) {
        this.dimension = dimension;
        this.input = input;
        setParents(input);
        int[][] shapes = new int[input.length][];
        for (int i = 0; i < input.length; i++) shapes[i] = input[i].getShape();
        setValue(DoubleTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return op(extractFromInputs(i -> input[i].getValue()));
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return null;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(extractFromInputs(i -> input[i].sample()));
    }

    protected DoubleTensor op(DoubleTensor... inputs) {
        DoubleTensor[] toConcat = Arrays.copyOfRange(inputs, 1, inputs.length);
        return inputs[0].concat(dimension, toConcat);
    }

    private DoubleTensor[] extractFromInputs(Function<Integer, DoubleTensor> func) {
        DoubleTensor[] extract = new DoubleTensor[input.length];
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(i);
        }
        return extract;
    }


}

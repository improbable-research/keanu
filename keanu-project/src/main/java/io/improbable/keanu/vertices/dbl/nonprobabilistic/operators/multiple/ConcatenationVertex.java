package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Arrays;
import java.util.Map;

public class ConcatenationVertex extends NonProbabilisticDouble {

    private final int dimension;
    private final DoubleVertex[] input;

    public ConcatenationVertex(int dimension, DoubleVertex... input) {
        this.dimension = dimension;
        this.input = input;
        setParents(input);
        //todo - calc shape
        setValue(DoubleTensor.placeHolder(new int[]{5, 4}));
    }

    @Override
    public DoubleTensor getDerivedValue() {
        DoubleTensor[] tensors = new DoubleTensor[input.length];
        for (int i = 0; i < input.length; i++) {
            tensors[i] = input[i].getValue();
        }
        DoubleTensor value = op(tensors);
        return value;
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return null;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        DoubleTensor[] tensors = new DoubleTensor[input.length];
        for (int i = 0; i < input.length; i++) {
            tensors[i] = input[i].sample(random);
        }
        return op(tensors);
    }

    protected DoubleTensor op(DoubleTensor... inputs) {
        DoubleTensor[] toConcat = Arrays.copyOfRange(inputs, 1, inputs.length);
        return inputs[0].concat(dimension, toConcat);
    }
}

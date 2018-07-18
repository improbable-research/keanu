package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.*;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

public class ConcatenationVertex extends NonProbabilisticDouble {

    private final int dimension;
    private final DoubleVertex[] input;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different.
     * @param input the input vertices to concatenate
     */
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
        Map<Long, List<DoubleTensor>> combinedPartialDerivativesOfInputs = new HashMap<>();

        for (DoubleVertex vertex : input) {
            for (Map.Entry<Long, DoubleTensor> partial : dualNumbers.get(vertex).getPartialDerivatives().asMap().entrySet()) {
                combinedPartialDerivativesOfInputs.computeIfAbsent(partial.getKey(), k -> new ArrayList<>()).add(partial.getValue());
            }
        }

        DualNumber dualOfPrimary = dualNumbers.get(input[0]);
        DoubleTensor[] inputValues = extractFromInputs(i -> input[i].getValue());
        DoubleTensor[] dualToConcat = Arrays.copyOfRange(inputValues, 1, inputValues.length);
        return dualOfPrimary.concat(dimension, combinedPartialDerivativesOfInputs, dualToConcat);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(extractFromInputs(i -> input[i].sample()));
    }

    protected DoubleTensor op(DoubleTensor... inputs) {
        DoubleTensor primary = inputs[0];
        DoubleTensor[] toConcat = Arrays.copyOfRange(inputs, 1, inputs.length);
        return primary.concat(dimension, toConcat);
    }

    private DoubleTensor[] extractFromInputs(Function<Integer, DoubleTensor> func) {
        DoubleTensor[] extract = new DoubleTensor[input.length];
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(i);
        }
        return extract;
    }


}

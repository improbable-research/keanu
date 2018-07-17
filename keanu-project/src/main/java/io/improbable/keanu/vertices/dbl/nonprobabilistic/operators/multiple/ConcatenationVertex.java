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
        Map<Long, DoubleTensor> concatDerivates = new HashMap<>();
        Map<Long, List<DoubleTensor>> partialDerivates = new HashMap<>();

        for (DoubleVertex vertex : input) {
            for (Map.Entry<Long, DoubleTensor> partial : dualNumbers.get(vertex).getPartialDerivatives().asMap().entrySet()) {
                partialDerivates.computeIfAbsent(partial.getKey(), k -> new ArrayList<>()).add(partial.getValue());
            }
        }

        for (Map.Entry<Long, List<DoubleTensor>> partial : partialDerivates.entrySet()) {
            concatDerivates.put(partial.getKey(), concatPartialDerivates(partial.getValue()));
        }

        return new DualNumber(dualNumbers.get(input[0]).getValue(), concatDerivates);
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

    private DoubleTensor concatPartialDerivates(List<DoubleTensor> partialDerivates) {
        if (partialDerivates.size() == 1) {
            return partialDerivates.get(0);
        } else {
            DoubleTensor primaryTensor = partialDerivates.remove(0);
            DoubleTensor[] derivativesToConcat = new DoubleTensor[partialDerivates.size()];
            return primaryTensor.concat(dimension, partialDerivates.toArray(derivativesToConcat));
        }
    }


}

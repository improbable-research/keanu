package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.lang.reflect.Array;
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
        super(v -> ((ConcatenationVertex) v).op(((ConcatenationVertex)v).extractFromInputs(DoubleTensor.class, Vertex::getValue)));
        this.dimension = dimension;
        this.input = input;
        setParents(input);
        int[][] shapes = extractFromInputs(int[].class, Vertex::getShape);
        setValue(DoubleTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        List<DualNumber> duals = new ArrayList<>();

        for (DoubleVertex vertex : input) {
            duals.add(dualNumbers.get(vertex));
        }

        DualNumber dualOfPrimary = duals.remove(0);
        DoubleTensor[] inputValues = extractFromInputs(DoubleTensor.class, Vertex::getValue);
        DoubleTensor[] dualToConcat = Arrays.copyOfRange(inputValues, 1, inputValues.length);
        return dualOfPrimary.concat(dimension, duals, dualToConcat);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(extractFromInputs(DoubleTensor.class, Vertex::sample));
    }

    protected DoubleTensor op(DoubleTensor... inputs) {
        DoubleTensor primary = inputs[0];
        DoubleTensor[] toConcat = Arrays.copyOfRange(inputs, 1, inputs.length);
        return primary.concat(dimension, toConcat);
    }

    private <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<DoubleTensor>, T> func) {
        T[] extract = (T[]) Array.newInstance(clazz, input.length);
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(input[i]);
        }
        return extract;
    }


}

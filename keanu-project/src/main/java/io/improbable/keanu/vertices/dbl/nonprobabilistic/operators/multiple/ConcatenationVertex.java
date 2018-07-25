package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class ConcatenationVertex extends DoubleVertex {

    private final int dimension;
    private final DoubleVertex[] input;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different.
     * @param input the input vertices to concatenate
     */
    public ConcatenationVertex(int dimension, DoubleVertex... input) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((ConcatenationVertex) v).applyConcat(Vertex::getValue)),
            Observable.observableTypeFor(ConcatenationVertex.class)
        );
        this.dimension = dimension;
        this.input = input;
        setParents(input);
        int[][] shapes = extractFromInputs(int[].class, Vertex::getShape);
        setValue(DoubleTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    private DoubleTensor applyConcat(Function<Vertex<DoubleTensor>, DoubleTensor> mapper) {
        DoubleTensor[] inputs = extractFromInputs(DoubleTensor.class, mapper);
        DoubleTensor primary = inputs[0];
        DoubleTensor[] toConcat = Arrays.copyOfRange(inputs, 1, inputs.length);
        return primary.concat(dimension, toConcat);
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        Map<Long, List<DoubleTensor>> combinedPartialDerivativesOfInputs = new HashMap<>();

        for (DoubleVertex vertex : input) {
            for (Map.Entry<Long, DoubleTensor> partial : dualNumbers.get(vertex).getPartialDerivatives().asMap().entrySet()) {
                combinedPartialDerivativesOfInputs.computeIfAbsent(partial.getKey(), k -> new ArrayList<>()).add(partial.getValue());
            }
        }

        DualNumber dualOfPrimary = dualNumbers.get(input[0]);
        DoubleTensor[] inputValues = extractFromInputs(DoubleTensor.class, Vertex::getValue);
        DoubleTensor[] dualToConcat = Arrays.copyOfRange(inputValues, 1, inputValues.length);
        return dualOfPrimary.concat(dimension, combinedPartialDerivativesOfInputs, dualToConcat);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return applyConcat(Vertex::sample);
    }

    private <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<DoubleTensor>, T> func) {
        T[] extract = (T[]) Array.newInstance(clazz, input.length);
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(input[i]);
        }
        return extract;
    }


}

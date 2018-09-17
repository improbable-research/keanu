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
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ConcatenationVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor> {

    private final int dimension;
    private final DoubleVertex[] input;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different.
     * @param input     the input vertices to concatenate
     */
    public ConcatenationVertex(int dimension, DoubleVertex... input) {
        this.dimension = dimension;
        this.input = input;
        setParents(input);
        int[][] shapes = extractFromInputs(int[].class, Vertex::getShape);
        setValue(DoubleTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        List<DualNumber> dualNumbersOfInputs = new ArrayList<>();

        for (DoubleVertex vertex : input) {
            dualNumbersOfInputs.add(dualNumbers.get(vertex));
        }
        DoubleTensor[] inputValues = extractFromInputs(DoubleTensor.class, Vertex::getValue);
        return DualNumber.concat(dualNumbers, dualNumbersOfInputs, input, dimension, inputValues);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> splitPartials = new HashMap<>();

        int currentSplitIndex = 0;
        int[] splitIndices = new int[input.length];

        for (int i = 0; i < input.length; i++) {
            splitIndices[i] = currentSplitIndex + input[i].getShape()[dimension];
            currentSplitIndex = splitIndices[i];
            splitPartials.put(input[i], new PartialDerivatives(new HashMap<>()));
        }

        int wrtDimensionToSliceOn = input[0].getShape().length + dimension;
        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
            DoubleTensor partial = entry.getValue();

            List<DoubleTensor> splitPartial = partial.split(wrtDimensionToSliceOn, splitIndices);

            for (int i = 0; i < splitPartial.size(); i++) {
                splitPartials.get(input[i]).putWithRespectTo(entry.getKey(), splitPartial.get(i));
            }

        }

        return splitPartials;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(extractFromInputs(DoubleTensor.class, Vertex::sample));
    }

    @Override
    public void calculate() {
        setValue(op(extractFromInputs(DoubleTensor.class, Vertex::getValue)));
    }

    protected DoubleTensor op(DoubleTensor... inputs) {
        return DoubleTensor.concat(dimension, inputs);
    }

    private <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<DoubleTensor>, T> func) {
        T[] extract = (T[]) Array.newInstance(clazz, input.length);
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(input[i]);
        }
        return extract;
    }

}

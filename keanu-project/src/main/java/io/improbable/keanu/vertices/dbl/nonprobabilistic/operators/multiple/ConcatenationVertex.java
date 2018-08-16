package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class ConcatenationVertex extends DoubleVertex implements Differentiable {

    private final int dimension;
    private final DoubleVertex[] input;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different.
     * @param input     the input vertices to concatenate
     */
    public ConcatenationVertex(int dimension, DoubleVertex... input) {
        super(new NonProbabilisticValueUpdater<>(
            v -> ((ConcatenationVertex) v).op(((ConcatenationVertex) v).extractFromInputs(DoubleTensor.class, Vertex::getValue))
        ));
        this.dimension = dimension;
        this.input = input;
        setParents(input);
        int[][] shapes = extractFromInputs(int[].class, Vertex::getShape);
        setValue(DoubleTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
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
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        DoubleTensor value = derivativeOfOutputsWithRespectToSelf.asMap().get(this.getId());
        int[] partialShape = value.getShape();
        int[] rearrange = TensorShape.dimensionRange(0, partialShape.length);
        rearrange[dimension] = 0;
        rearrange[0] = dimension;

        DoubleTensor permuted = value.permute(rearrange);
        double[] permutedBuffer = permuted.asFlatDoubleArray();

        Map<Vertex, PartialDerivatives> concattedPartial = new HashMap<>();

        int bufferOffset = 0;
        for (DoubleVertex vertex : input) {
            int[] ofWrtShape = TensorShape.concat(Arrays.copyOfRange(value.getShape(), 0, vertex.getValue().getRank()), vertex.getShape());
            int inputSize = (int) (value.getLength() / (value.getShape()[value.getShape().length / 2 + dimension])) * vertex.getShape()[dimension];
            double[] inputsDualNumbers = Arrays.copyOfRange(permutedBuffer, bufferOffset, bufferOffset + inputSize);
            DoubleTensor unpermuted = DoubleTensor.create(inputsDualNumbers, ofWrtShape).permute(rearrange);
            PartialDerivatives partial = new PartialDerivatives(getId(), unpermuted);
            concattedPartial.put(vertex, partial);
            bufferOffset += inputSize;
        }

        return concattedPartial;
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

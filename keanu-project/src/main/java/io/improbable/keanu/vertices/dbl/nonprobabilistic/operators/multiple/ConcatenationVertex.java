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
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
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

        //TODO: Make this work.
        //Wrt self produces an identity matrix, which we can flatten.
        //Eg a concat of a 2x2 and 2x2 on dimension zero to make a 4x2. Produces a 4x2x4x2.
        //Flatten this to an 8x8 identity matrix.
        //If we slice the identity matrix halfway down the rows to make a 4x8.
        //10000000
        //01000000
        //00100000
        //00010000
        //Then read downwards along columns we produce the correct results.
        //Why? I don't know. Oh god.

        DoubleTensor value = derivativeOfOutputsWithRespectToSelf.asMap().get(this.getId());
        int[] partialShape = value.getShape();
        int[] rearrange = TensorShape.dimensionRange(0, partialShape.length);
        rearrange[dimension] = 0;
        rearrange[0] = dimension;
        //Doing a permute of (3, 1, 0, 2) works. Why?!

        DoubleTensor permuted = value.permute(rearrange);
        double[] permutedBuffer = permuted.asFlatDoubleArray();

        Map<Vertex, PartialDerivatives> concattedPartial = new HashMap<>();

        int bufferOffset = 0;
        for (DoubleVertex vertex : input) {
            int[] ofWrtShape = TensorShape.concat(Arrays.copyOfRange(value.getShape(), 0, vertex.getValue().getRank()), vertex.getShape());
            int inputSize = (int) (value.getLength() / (value.getShape()[value.getShape().length / 2 + dimension])) * vertex.getShape()[dimension];
            double[] inputsDualNumbers = Arrays.copyOfRange(permutedBuffer, bufferOffset, bufferOffset + inputSize);

            DoubleTensor bufferExtracted = DoubleTensor.create(inputsDualNumbers, ofWrtShape);
            //Not currently using the buffer as permute is not producing the correct numbers.
            //How do we go from wrtSelf to the correct split partials?

            DoubleTensor alongDimension = extractAlongDimension(permuted, vertex, input);
            PartialDerivatives partial = new PartialDerivatives(getId(), alongDimension);
            concattedPartial.put(vertex, partial);
            bufferOffset += inputSize;
        }

        return concattedPartial;
    }

    private DoubleTensor extractAlongDimension(DoubleTensor buffer, DoubleVertex input, DoubleVertex[] inputs) {
        int[] sizeOfInput = new int[input.getShape()[dimension]];
        int count = 0;
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] != input) {
                count += input.getShape()[dimension];
            } else {
                sizeOfInput = TensorShape.dimensionRange(count, count + input.getShape()[dimension]);
            }
        }
        DoubleTensor slice = buffer.slice(input.getShape()[dimension], sizeOfInput);
        DoubleTensor reshaped = slice.reshape(TensorShape.concat(getShape(), input.getShape()));
        return reshaped;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(extractFromInputs(DoubleTensor.class, Vertex::sample));
    }

    @Override
    public DoubleTensor calculate() {
        return op(extractFromInputs(DoubleTensor.class, Vertex::getValue));
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

package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

public class BoolConcatenationVertex extends BoolVertex implements NonProbabilistic<BooleanTensor> {

    private static final String DIMENSION_NAME = "dimension";
    private static final String INPUT_NAME = "inputArray";
    private final int dimension;
    private final BoolVertex[] input;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different.
     * @param input     the input vertices to concatenate
     */
    public BoolConcatenationVertex(int dimension, BoolVertex... input) {
        super(checkShapesCanBeConcatenated(dimension, extractFromInputs(long[].class, Vertex::getShape, input)));
        this.dimension = dimension;
        this.input = input;
        setParents(input);
    }

    public BoolConcatenationVertex(@LoadVertexParam(DIMENSION_NAME) int dimension,
                                   @LoadVertexParam(INPUT_NAME) Vertex[] input) {
        this(dimension, convertVertexArrayToBoolVertex(input));
    }

    private static BoolVertex[] convertVertexArrayToBoolVertex(Vertex[] input) {
        return Arrays.stream(input).toArray(BoolVertex[]::new);
    }

    @Override
    public BooleanTensor calculate() {
        return op(extractFromInputs(BooleanTensor.class, Vertex::getValue, input));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(extractFromInputs(BooleanTensor.class, Vertex::sample, input));
    }

    protected BooleanTensor op(BooleanTensor... inputs) {
        return BooleanTensor.concat(dimension, inputs);
    }

    private static <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<BooleanTensor>, T> func, BoolVertex[] input) {
        T[] extract = (T[]) Array.newInstance(clazz, input.length);
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(input[i]);
        }
        return extract;
    }

    @SaveVertexParam(DIMENSION_NAME)
    public int getDimensionParam() {
        return dimension;
    }

    @SaveVertexParam(INPUT_NAME)
    public BoolVertex[] getInputParam() {
        return input;
    }
}

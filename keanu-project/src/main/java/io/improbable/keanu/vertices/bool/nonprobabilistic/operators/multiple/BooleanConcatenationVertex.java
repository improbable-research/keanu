package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.BooleanVertex;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

public class BooleanConcatenationVertex extends VertexImpl<BooleanTensor, BooleanVertex> implements BooleanVertex, NonProbabilistic<BooleanTensor> {

    private static final String DIMENSION_NAME = "dimension";
    private static final String OPERANDS_NAME = "operands";
    private final int dimension;
    private final BooleanVertex[] operands;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different.
     * @param operands  the input vertices to concatenate
     */
    public BooleanConcatenationVertex(int dimension, BooleanVertex... operands) {
        super(checkShapesCanBeConcatenated(dimension, extractFromInputs(long[].class, Vertex::getShape, operands)));
        this.dimension = dimension;
        this.operands = operands;
        setParents(operands);
    }

    @ExportVertexToPythonBindings
    public BooleanConcatenationVertex(@LoadVertexParam(DIMENSION_NAME) int dimension,
                                      @LoadVertexParam(OPERANDS_NAME) Vertex[] input) {
        this(dimension, convertVertexArrayToBooleanVertex(input));
    }

    private static BooleanVertex[] convertVertexArrayToBooleanVertex(Vertex[] input) {
        return Arrays.stream(input).toArray(BooleanVertex[]::new);
    }

    @Override
    public BooleanTensor calculate() {
        return op(extractFromInputs(BooleanTensor.class, Vertex::getValue, operands));
    }

    protected BooleanTensor op(BooleanTensor... inputs) {
        return BooleanTensor.concat(dimension, inputs);
    }

    private static <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<BooleanTensor, BooleanVertex>, T> func, BooleanVertex[] input) {
        T[] extract = (T[]) Array.newInstance(clazz, input.length);
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(input[i]);
        }
        return extract;
    }

    @SaveVertexParam(DIMENSION_NAME)
    public int getDimension() {
        return dimension;
    }

    @SaveVertexParam(OPERANDS_NAME)
    public BooleanVertex[] getOperands() {
        return operands;
    }
}

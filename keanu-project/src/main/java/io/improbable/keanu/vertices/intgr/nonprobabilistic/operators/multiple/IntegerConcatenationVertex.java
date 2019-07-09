package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.multiple;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

public class IntegerConcatenationVertex extends VertexImpl<IntegerTensor> implements IntegerVertex, NonProbabilistic<IntegerTensor> {

    private final static String DIMENSION_NAME = "dimension";
    private final static String OPERANDS_NAME = "operands";

    private final int dimension;
    private final IntegerVertex[] operands;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes may be different.
     * @param operands  the input vertices to concatenate
     */
    public IntegerConcatenationVertex(int dimension, IntegerVertex... operands) {
        super(checkShapesCanBeConcatenated(dimension, extractFromInputs(long[].class, Vertex::getShape, operands)));
        this.dimension = dimension;
        this.operands = operands;
        setParents(operands);
    }

    @ExportVertexToPythonBindings
    public IntegerConcatenationVertex(@LoadVertexParam(DIMENSION_NAME) int dimension,
                                      @LoadVertexParam(OPERANDS_NAME) Vertex[] input) {
        this(dimension, convertVertexArrayToIntegerVertex(input));
    }

    private static IntegerVertex[] convertVertexArrayToIntegerVertex(Vertex[] input) {
        return Arrays.stream(input).toArray(IntegerVertex[]::new);
    }

    @Override
    public IntegerTensor calculate() {
        return op(extractFromInputs(IntegerTensor.class, Vertex::getValue, operands));
    }

    private IntegerTensor op(IntegerTensor... inputs) {
        return IntegerTensor.concat(dimension, inputs);
    }

    private static <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<IntegerTensor>, T> func, IntegerVertex[] input) {
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
    public IntegerVertex[] getOperands() {
        return operands;
    }
}

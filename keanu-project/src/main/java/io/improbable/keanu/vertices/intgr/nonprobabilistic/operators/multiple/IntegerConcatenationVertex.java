package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.multiple;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesCanBeConcatenated;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import java.lang.reflect.Array;
import java.util.function.Function;

public class IntegerConcatenationVertex extends IntegerVertex
        implements NonProbabilistic<IntegerTensor> {

    private final int dimension;
    private final IntegerVertex[] input;

    /**
     * A vertex that can concatenate any amount of vertices along a given dimension.
     *
     * @param dimension the dimension to concatenate on. This is the only dimension in which sizes
     *     may be different.
     * @param input the input vertices to concatenate
     */
    public IntegerConcatenationVertex(int dimension, IntegerVertex... input) {
        this.dimension = dimension;
        this.input = input;
        setParents(input);
        int[][] shapes = extractFromInputs(int[].class, Vertex::getShape);
        setValue(IntegerTensor.placeHolder(checkShapesCanBeConcatenated(dimension, shapes)));
    }

    @Override
    public IntegerTensor calculate() {
        return op(extractFromInputs(IntegerTensor.class, Vertex::getValue));
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op(extractFromInputs(IntegerTensor.class, Vertex::sample));
    }

    private IntegerTensor op(IntegerTensor... inputs) {
        return IntegerTensor.concat(dimension, inputs);
    }

    private <T> T[] extractFromInputs(Class<T> clazz, Function<Vertex<IntegerTensor>, T> func) {
        T[] extract = (T[]) Array.newInstance(clazz, input.length);
        for (int i = 0; i < input.length; i++) {
            extract[i] = func.apply(input[i]);
        }
        return extract;
    }
}

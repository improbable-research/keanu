package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShape.removeDimension;

public class SliceVertex extends DoubleUnaryOpVertex implements Differentiable {

    private final int dimension;
    private final long index;
    private final static String DIMENSION_NAME = "dimension";
    private final static String INDEX_NAME = "index";

    /**
     * Takes the slice along a given dimension and index of a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension   the dimension to extract along
     * @param index       the index of extraction
     */
    public SliceVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex,
                       @LoadVertexParam(DIMENSION_NAME) int dimension,
                       @LoadVertexParam(INDEX_NAME) long index) {
        super(removeDimension(dimension, inputVertex.getShape()), inputVertex);
        this.dimension = dimension;
        this.index = index;
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.slice(dimension, index);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();

        DoubleTensor partial = derivativeOfOutputWithRespectToSelf.get();
        DoubleTensor padded = padSliceWithZerosToMatchOriginalShape(partial);
        partials.put(inputVertex, new PartialDerivative(padded));

        return partials;
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative dInputVertex = derivativeOfParentsWithRespectToInput.get(inputVertex);
        return new PartialDerivative(dInputVertex.get().slice(dimension, index));
    }

    private DoubleTensor padSliceWithZerosToMatchOriginalShape(DoubleTensor tensor) {
        long[] targetShape = TensorShape.concat(getShape(), inputVertex.getShape());
        int dimensionInWrt = dimension + getRank();
        long indicesBefore = index;
        long indicesAfter = targetShape[dimensionInWrt] - index - 1;
        targetShape[dimensionInWrt] = 1;
        DoubleTensor outputTensor = tensor.reshape(targetShape);

        if (indicesBefore != 0) {
            targetShape[dimensionInWrt] = indicesBefore;
            DoubleTensor prefixTensor = DoubleTensor.zeros(targetShape).reshape(targetShape);
            outputTensor = DoubleTensor.concat(dimensionInWrt, prefixTensor, outputTensor);
        }

        if (indicesAfter != 0) {
            targetShape[dimensionInWrt] = indicesAfter;
            DoubleTensor postfixTensor = DoubleTensor.zeros(targetShape).reshape(targetShape);
            outputTensor = DoubleTensor.concat(dimensionInWrt, outputTensor, postfixTensor);
        }

        return outputTensor;
    }

    @SaveVertexParam(DIMENSION_NAME)
    public int getDimension() {
        return dimension;
    }

    @SaveVertexParam(INDEX_NAME)
    public long getIndex() {
        return index;
    }
}

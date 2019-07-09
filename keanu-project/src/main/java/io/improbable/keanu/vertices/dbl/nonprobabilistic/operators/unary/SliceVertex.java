package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
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
    @ExportVertexToPythonBindings
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
    public Map<IVertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<IVertex, PartialDerivative> partials = new HashMap<>();

        DoubleTensor partial = derivativeOfOutputWithRespectToSelf.get();
        DoubleTensor padded = padSliceWithZerosToMatchInputShape(partial);
        partials.put(inputVertex, new PartialDerivative(padded));

        return partials;
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<IVertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative dInputVertex = derivativeOfParentsWithRespectToInput.get(inputVertex);
        return new PartialDerivative(dInputVertex.get().slice(dimension, index));
    }

    private DoubleTensor padSliceWithZerosToMatchInputShape(DoubleTensor tensor) {
        int dimensionsInWrt = getRank();
        int dimensionsInOf = tensor.getRank() - dimensionsInWrt;
        int sliceDimension = dimension + dimensionsInOf;
        long[] targetShape = TensorShape.concat(
            TensorShape.selectDimensions(0, dimensionsInOf, tensor.getShape()),
            inputVertex.getShape()
        );
        long indicesBefore = index;
        long indicesAfter = targetShape[sliceDimension] - index - 1;
        targetShape[sliceDimension] = 1;

        DoubleTensor outputTensor = tensor.reshape(targetShape);

        if (indicesBefore != 0) {
            targetShape[sliceDimension] = indicesBefore;
            DoubleTensor prefixTensor = DoubleTensor.zeros(targetShape);
            outputTensor = DoubleTensor.concat(sliceDimension, prefixTensor, outputTensor);
        }

        if (indicesAfter != 0) {
            targetShape[sliceDimension] = indicesAfter;
            DoubleTensor postfixTensor = DoubleTensor.zeros(targetShape);
            outputTensor = DoubleTensor.concat(sliceDimension, outputTensor, postfixTensor);
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

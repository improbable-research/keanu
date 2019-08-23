package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShape.removeDimension;

public class SliceVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private final static String DIMENSION_NAME = "dimension";
    private final static String INDEX_NAME = "index";

    private final int dimension;
    private final long index;

    /**
     * Takes the slice along a given dimension and index of a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension   the dimension to extract along
     * @param index       the index of extraction
     */
    @ExportVertexToPythonBindings
    public SliceVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                       @LoadVertexParam(DIMENSION_NAME) int dimension,
                       @LoadVertexParam(INDEX_NAME) long index) {
        super(removeDimension(dimension, inputVertex.getShape()), inputVertex, inputVertex.ofType());
        this.dimension = dimension;
        this.index = index;
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.slice(dimension, index);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();

        DoubleTensor partial = derivativeOfOutputWithRespectToSelf.get();
        DoubleTensor padded = padSliceWithZerosToMatchInputShape(partial);
        partials.put(inputVertex, new PartialDerivative(padded));

        return partials;
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        ForwardModePartialDerivative dInputVertex = derivativeOfParentsWithRespectToInput.get(inputVertex);
        return new ForwardModePartialDerivative(dInputVertex.getWrtShape(), dInputVertex.get().slice(dimension + dInputVertex.getWrtShape().length, index));
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

    @Override
    public boolean isDifferentiable() {
        return inputVertex.isDifferentiable();
    }

    @SaveVertexParam(DIMENSION_NAME)
    public int getDimension() {
        return this.dimension;
    }

    @SaveVertexParam(INDEX_NAME)
    public long getIndex() {
        return this.index;
    }
}

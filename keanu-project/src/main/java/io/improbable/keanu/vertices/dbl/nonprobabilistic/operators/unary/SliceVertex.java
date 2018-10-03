package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.tensor.TensorShape.shapeSlice;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class SliceVertex extends DoubleUnaryOpVertex {

    private final int dimension;
    private final int index;

    /**
     * Takes the slice along a given dimension and index of a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension   the dimension to extract along
     * @param index       the index of extraction
     */
    public SliceVertex(DoubleVertex inputVertex, int dimension, int index) {
        super(shapeSlice(dimension, inputVertex.getShape()), inputVertex);
        this.dimension = dimension;
        this.index = index;
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.slice(dimension, index);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue();
            DoubleTensor padded = padSliceWithZerosToMatchOriginalShape(v);
            partials.put(inputVertex, new PartialDerivatives(k, padded));
        }

        return partials;
     }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        return dualNumber.slice(dimension, index);
    }

    private DoubleTensor padSliceWithZerosToMatchOriginalShape(DoubleTensor tensor) {
        int[] targetShape = TensorShape.concat(getShape(), inputVertex.getShape());
        int dimensionInWrt = dimension + getShape().length;
        int indicesBefore = index;
        int indicesAfter = targetShape[dimensionInWrt] - index - 1;
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

}

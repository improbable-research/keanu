package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class TakeVertex extends DoubleUnaryOpVertex {

    private final int[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex to extract from
     * @param index       the index to extract at
     */
    public TakeVertex(DoubleVertex inputVertex, int... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
        this.index = index;
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return DoubleTensor.scalar(value.getValue(index));
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives derivativeOfParentWithRespectToInputs) {

        Map<VertexId, DoubleTensor> partialsOf = new HashMap<>();
        DoubleTensor newValue = this.getValue();

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeOfParentWithRespectToInputs.asMap().entrySet()) {
            DoubleTensor atIndexTensor = takeFromPartial(entry.getValue(), index);
            int desiredRank = atIndexTensor.getShape().length + newValue.getShape().length;
            long[] paddedShape = TensorShape.shapeToDesiredRankByPrependingOnes(atIndexTensor.getShape(), desiredRank);
            atIndexTensor = atIndexTensor.reshape(paddedShape);
            partialsOf.put(entry.getKey(), atIndexTensor);
        }

        return new PartialDerivatives(partialsOf);
    }

    private DoubleTensor takeFromPartial(DoubleTensor from, int... indices) {
        long[] fromShape = from.getShape();
        long[] subFromShape = Arrays.copyOf(fromShape, indices.length);
        int indexToTakeFrom = TensorShape.getFlatIndex(subFromShape, TensorShape.getRowFirstStride(subFromShape), indices);
        long[] takeShape = Arrays.copyOfRange(fromShape, indices.length, fromShape.length);
        int subShapeLength = (int) TensorShape.getLength(subFromShape);

        return from.reshape(subShapeLength, -1)
            .slice(0, indexToTakeFrom)
            .reshape(takeShape);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> reshapedDerivatives = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partialDerivative : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
            DoubleTensor partial = partialDerivative.getValue();
            long[] newPartialShape = TensorShape.concat(
                TensorShape.selectDimensions(0, partial.getRank() - getShape().length, partial.getShape()),
                inputVertex.getShape()
            );
            DoubleTensor highRankZeros = DoubleTensor.zeros(newPartialShape);
            long[] partialUpRankShape = TensorShape.shapeDesiredToRankByAppendingOnes(partial.getShape(), newPartialShape.length);
            DoubleTensor partialBroadcastToHighRank = highRankZeros.plus(partial.reshape(partialUpRankShape));
            DoubleTensor takeMask = DoubleTensor.zeros(inputVertex.getShape()).setValue(1., index);
            DoubleTensor highRankMask = partialBroadcastToHighRank.times(takeMask);
            reshapedDerivatives.put(inputVertex, new PartialDerivatives(partialDerivative.getKey(), highRankMask));
        }

        return reshapedDerivatives;
    }
}

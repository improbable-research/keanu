package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class TakeVertex extends DoubleUnaryOpVertex implements Differentiable {

    private static final String INDEX_NAME = "index";
    private final long[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex to extract from
     * @param index       the index to extract at
     */
    public TakeVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex,
                      @LoadVertexParam(INDEX_NAME) long... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
        this.index = index;
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return DoubleTensor.scalar(value.getValue(index));
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);

        DoubleTensor newValue = this.getValue();

        DoubleTensor atIndexTensor = takeFromPartial(derivativeOfParentWithRespectToInputs.getPartial(), index);
        int desiredRank = atIndexTensor.getShape().length + newValue.getShape().length;
        long[] paddedShape = TensorShape.shapeToDesiredRankByPrependingOnes(atIndexTensor.getShape(), desiredRank);
        atIndexTensor = atIndexTensor.reshape(paddedShape);

        return new PartialDerivative(derivativeOfParentWithRespectToInputs.getKey(), atIndexTensor);
    }

    private DoubleTensor takeFromPartial(DoubleTensor from, long... indices) {
        long[] fromShape = from.getShape();
        long[] subFromShape = Arrays.copyOf(fromShape, indices.length);
        long indexToTakeFrom = TensorShape.getFlatIndex(subFromShape, TensorShape.getRowFirstStride(subFromShape), indices);
        long[] takeShape = Arrays.copyOfRange(fromShape, indices.length, fromShape.length);
        long subShapeLength = TensorShape.getLength(subFromShape);

        return from.reshape(subShapeLength, -1)
            .slice(0, indexToTakeFrom)
            .reshape(takeShape);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> reshapedDerivatives = new HashMap<>();

        DoubleTensor partial = derivativeOfOutputWithRespectToSelf.getPartial();
        long[] newPartialShape = TensorShape.concat(
            TensorShape.selectDimensions(0, partial.getRank() - getShape().length, partial.getShape()),
            inputVertex.getShape()
        );
        DoubleTensor highRankZeros = DoubleTensor.zeros(newPartialShape);
        long[] partialUpRankShape = TensorShape.shapeDesiredToRankByAppendingOnes(partial.getShape(), newPartialShape.length);
        DoubleTensor partialBroadcastToHighRank = highRankZeros.plus(partial.reshape(partialUpRankShape));
        DoubleTensor takeMask = DoubleTensor.zeros(inputVertex.getShape()).setValue(1., index);
        DoubleTensor highRankMask = partialBroadcastToHighRank.times(takeMask);
        reshapedDerivatives.put(inputVertex, new PartialDerivative(derivativeOfOutputWithRespectToSelf.getKey(), highRankMask));

        return reshapedDerivatives;
    }

    @SaveVertexParam(INDEX_NAME)
    public long[] getIndex() {
        return index;
    }
}

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
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class TakeVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String INDEX = "index";
    private long[] index;

    @ExportVertexToPythonBindings
    public TakeVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                      @LoadVertexParam(INDEX) long... index) {
        super(new long[0], inputVertex, inputVertex.ofType());
        this.index = index;
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);

        TENSOR newValue = this.getValue();

        DoubleTensor atIndexTensor = takeFromPartial(derivativeOfParentWithRespectToInputs.get(), index);
        int desiredRank = atIndexTensor.getRank() + newValue.getRank();
        long[] paddedShape = TensorShape.shapeToDesiredRankByPrependingOnes(atIndexTensor.getShape(), desiredRank);
        atIndexTensor = atIndexTensor.reshape(paddedShape);

        return new PartialDerivative(atIndexTensor);
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

        DoubleTensor partial = derivativeOfOutputWithRespectToSelf.get();
        long[] newPartialShape = TensorShape.concat(
            TensorShape.selectDimensions(0, partial.getRank() - getRank(), partial.getShape()),
            inputVertex.getShape()
        );
        long[] partialUpRankShape = TensorShape.shapeDesiredToRankByAppendingOnes(partial.getShape(), newPartialShape.length);
        DoubleTensor partialBroadcastToHighRank = partial.reshape(partialUpRankShape).broadcast(newPartialShape);
        DoubleTensor takeMask = DoubleTensor.zeros(inputVertex.getShape());
        takeMask.setValue(1., index);
        DoubleTensor highRankMask = partialBroadcastToHighRank.times(takeMask);
        reshapedDerivatives.put(inputVertex, new PartialDerivative(highRankMask));

        return reshapedDerivatives;
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.take(index);
    }

    @Override
    public boolean isDifferentiable() {
        return inputVertex.isDifferentiable();
    }

    @SaveVertexParam(INDEX)
    public long[] getIndex() {
        return this.index;
    }
}
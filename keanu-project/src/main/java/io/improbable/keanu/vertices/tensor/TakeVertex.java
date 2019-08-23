package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

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
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        ForwardModePartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);

        TENSOR newValue = this.getValue();

        DoubleTensor atIndexTensor = takeFromPartial(derivativeOfParentWithRespectToInputs.get(), index);
        int desiredRank = atIndexTensor.getRank() + newValue.getRank();
        long[] paddedShape = TensorShape.shapeToDesiredRankByPrependingOnes(atIndexTensor.getShape(), desiredRank);
        atIndexTensor = atIndexTensor.reshape(paddedShape);

        return new ForwardModePartialDerivative(derivativeOfParentWithRespectToInputs.getWrtShape(), atIndexTensor);
    }

    private DoubleTensor takeFromPartial(DoubleTensor from, long... indices) {
        Slicer.SlicerBuilder builder = Slicer.builder().ellipsis();

        for (long i : indices) {
            builder.slice(i);
        }

        return from.slice(
            builder.build()
        );
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, ReverseModePartialDerivative> reshapedDerivatives = new HashMap<>();

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
        reshapedDerivatives.put(inputVertex, new ReverseModePartialDerivative(derivativeOfOutputWithRespectToSelf.getOfShape(), highRankMask));

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
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
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.AutoDiffBroadcast.broadcastPartialReverse;

public class BroadcastVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String TO_SHAPE = "toShape";
    private final long[] toShape;

    @ExportVertexToPythonBindings
    public BroadcastVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                           @LoadVertexParam(TO_SHAPE) long... toShape) {
        super(toShape, inputVertex, inputVertex.ofType());
        this.toShape = toShape;
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.broadcast(toShape);
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {

        final ForwardModePartialDerivative dInput = derivativeOfParentsWithRespectToInput.get(inputVertex);

        final long[] partialOfShape = inputVertex.getShape();
        final long[] wrtShape = dInput.getWrtShape();
        final long[] partialReshape = TensorShape.concat(wrtShape, TensorShape.shapeToDesiredRankByPrependingOnes(partialOfShape, toShape.length));
        final long[] resultShape = TensorShape.concat(wrtShape, toShape);

        final DoubleTensor correctedPartial = dInput.get().reshape(partialReshape).broadcast(resultShape);

        return new ForwardModePartialDerivative(wrtShape, correctedPartial);
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative partial) {

        Map<Vertex, ReverseModePartialDerivative> result = new HashMap<>();
        result.put(inputVertex, broadcastPartialReverse(partial, getShape(), inputVertex.getShape()));
        return result;
    }

    @SaveVertexParam(TO_SHAPE)
    public long[] getToShape() {
        return this.toShape;
    }
}

package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.AutoDiffBroadcast.broadcastPartialForward;
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

        ForwardModePartialDerivative dInput = derivativeOfParentsWithRespectToInput.get(inputVertex);
        return broadcastPartialForward(dInput, inputVertex.getShape(), toShape);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {

        Map<Vertex, PartialDerivative> result = new HashMap<>();
        result.put(inputVertex, broadcastPartialReverse(partial, getShape(), inputVertex.getShape()));
        return result;
    }

    @SaveVertexParam(TO_SHAPE)
    public long[] getToShape() {
        return this.toShape;
    }
}

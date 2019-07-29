package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.AutoDiffBroadcast.dimensionsWithShapeChange;

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
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {

        PartialDerivative dInput = derivativeOfParentsWithRespectToInput.get(inputVertex);
        DoubleTensor dThis = dInput.get().broadcast(TensorShape.concat(toShape, dInput.getWrtShape(inputVertex.getShape())));

        return new PartialDerivative(dThis);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {

        long[] partialShape = partial.get().getShape();
        long[] partialWrtShape = getShape();
        long[] targetWrtShape = inputVertex.getShape();

        int[] broadcastDimensions = dimensionsWithShapeChange(partialShape, partialWrtShape.length, targetWrtShape);

        DoubleTensor partialSummed = partial.get().sum(broadcastDimensions);

        long[] resultShape = TensorShape.concat(
            partial.getOfShape(partialWrtShape),
            targetWrtShape
        );

        Map<Vertex, PartialDerivative> result = new HashMap<>();
        result.put(inputVertex, new PartialDerivative(partialSummed.reshape(resultShape)));
        return result;
    }

    @SaveVertexParam(TO_SHAPE)
    public long[] getToShape() {
        return this.toShape;
    }
}

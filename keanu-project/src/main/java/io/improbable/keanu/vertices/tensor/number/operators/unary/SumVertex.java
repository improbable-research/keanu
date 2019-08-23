package io.improbable.keanu.vertices.tensor.number.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.UnaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShape.getReductionResultShape;
import static java.util.Collections.singletonMap;

public class SumVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String DIMENSIONS_NAME = "overDimensions";
    private final int[] overDimensions;

    /**
     * Performs a sum across specified dimensions. Negative dimension indexing is not supported.
     *
     * @param inputVertex    the vertex to have its values summed
     * @param overDimensions dimensions to sum over
     */
    @ExportVertexToPythonBindings
    public SumVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                     @LoadVertexParam(DIMENSIONS_NAME) int[] overDimensions) {
        super(getReductionResultShape(inputVertex.getShape(), overDimensions), inputVertex, inputVertex.ofType());
        this.overDimensions = overDimensions;
    }

    /**
     * Performs a sum across all dimensions
     *
     * @param inputVertex the vertex to have its values summed
     */
    public SumVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(Tensor.SCALAR_SHAPE, inputVertex, inputVertex.ofType());
        this.overDimensions = null;
    }

    @Override
    protected TENSOR op(TENSOR value) {
        if (overDimensions == null) {
            return value.sum();
        } else {
            return value.sum(overDimensions);
        }
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        final ForwardModePartialDerivative dInputVertex = derivativeOfParentsWithRespectToInput.get(inputVertex);
        final int operandRank = inputVertex.getValue().getRank();
        final int partialRank = dInputVertex.get().getRank();

        final int[] dimensionsToSum;
        if (overDimensions == null) {
            dimensionsToSum = TensorShape.dimensionRange(operandRank, partialRank);
        } else {
            dimensionsToSum = new int[overDimensions.length];
            for (int i = 0; i < dimensionsToSum.length; i++) {
                dimensionsToSum[i] = overDimensions[i] + (partialRank - operandRank);
            }
        }

        return new ForwardModePartialDerivative(dInputVertex.getWrtShape(), dInputVertex.get().sum(dimensionsToSum));
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {

        final long[] wrtShapeWithoutRankLoss = TensorShape.getReductionResultShapeWithoutRankLoss(inputVertex.getShape(), overDimensions);
        final long[] ofShape = partial.getOfShape();

        final long[] newPartialShape = TensorShape.concat(
            ofShape,
            wrtShapeWithoutRankLoss
        );

        final DoubleTensor partialDueToSummationShapeChange = partial.get().reshape(newPartialShape);

        final long[] resultShape = TensorShape.concat(
            ofShape,
            inputVertex.getShape()
        );

        final DoubleTensor broadcastedPartial = partialDueToSummationShapeChange.broadcast(resultShape);

        return singletonMap(inputVertex, new PartialDerivative(ofShape, broadcastedPartial));
    }

    @SaveVertexParam(DIMENSIONS_NAME)
    public int[] getOverDimensions() {
        return overDimensions;
    }
}

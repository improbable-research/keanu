package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.FloatingPointTensor;
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
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShape.getReductionResultShape;
import static java.util.Collections.singletonMap;

public class MeanVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
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
    public MeanVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                      @LoadVertexParam(DIMENSIONS_NAME) int[] overDimensions) {
        super(getReductionResultShape(inputVertex.getShape(), overDimensions), inputVertex, inputVertex.ofType());
        this.overDimensions = overDimensions;
    }

    /**
     * Performs a sum across all dimensions
     *
     * @param inputVertex the vertex to have its values summed
     */
    public MeanVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(Tensor.SCALAR_SHAPE, inputVertex, inputVertex.ofType());
        this.overDimensions = null;
    }

    @Override
    protected TENSOR op(TENSOR value) {
        if (overDimensions == null) {
            return value.mean();
        } else {
            return value.mean(overDimensions);
        }
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        final PartialDerivative dInputVertex = derivativeOfParentsWithRespectToInput.get(inputVertex);
        final int operandRank = inputVertex.getValue().getRank();
        final int[] dimensionsToSum = overDimensions == null ? TensorShape.dimensionRange(0, operandRank) : overDimensions;

        final long length = TensorShape.getLength(inputVertex.getShape(), dimensionsToSum);

        return new PartialDerivative(dInputVertex.get().sum(dimensionsToSum).div(length));
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {
        final long[] wrtShapeWithoutRankLoss = TensorShape.getReductionResultShapeWithoutRankLoss(inputVertex.getShape(), overDimensions);
        final long[] ofShape = partial.getOfShape(this.getShape());

        final long[] newPartialShape = TensorShape.concat(
            ofShape,
            wrtShapeWithoutRankLoss
        );

        final DoubleTensor partialDueToReductionShapeChange = partial.get().reshape(newPartialShape);

        final long[] resultShape = TensorShape.concat(
            ofShape,
            inputVertex.getShape()
        );

        final long length = overDimensions == null ? inputVertex.getLength() : TensorShape.getLength(inputVertex.getShape(), overDimensions);

        final DoubleTensor broadcastedPartial = partialDueToReductionShapeChange.broadcast(resultShape).div(length);

        return singletonMap(inputVertex, new PartialDerivative(broadcastedPartial));
    }

    @SaveVertexParam(DIMENSIONS_NAME)
    public int[] getOverDimensions() {
        return overDimensions;
    }
}
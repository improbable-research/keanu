package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import com.google.common.base.Preconditions;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.Map;

import static java.util.Collections.singletonMap;

public class SumVertex extends DoubleUnaryOpVertex implements Differentiable {

    private static final String DIMENSIONS_NAME = "overDimensions";
    private final int[] overDimensions;

    /**
     * Performs a sum across specified dimensions. Negative dimension indexing is not supported.
     *
     * @param inputVertex    the vertex to have its values summed
     * @param overDimensions dimensions to sum over
     */
    public SumVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex,
                     @LoadVertexParam(DIMENSIONS_NAME) int[] overDimensions) {
        super(getSummationResultShape(inputVertex.getShape(), overDimensions), inputVertex);
        this.overDimensions = overDimensions;
    }

    /**
     * Performs a sum across all dimensions
     *
     * @param inputVertex the vertex to have its values summed
     */
    @ExportVertexToPythonBindings
    public SumVertex(DoubleVertex inputVertex) {
        this(inputVertex, TensorShape.dimensionRange(0, inputVertex.getRank()));
    }

    private static long[] getSummationResultShape(long[] inputShape, int[] sumOverDimensions) {
        if (inputShape.length > 0) {
            return ArrayUtils.removeAll(inputShape, sumOverDimensions);
        } else {
            Preconditions.checkArgument(sumOverDimensions.length == 0);
            return inputShape;
        }
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.sum(overDimensions);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative dInputVertex = derivativeOfParentsWithRespectToInput.get(inputVertex);
        return new PartialDerivative(dInputVertex.get().sum(overDimensions));
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {

        long[] wrtShapeWithoutRankLoss = summedOverShapeWithoutRankLoss(inputVertex.getShape(), overDimensions);
        long[] ofShape = derivativeOfOutputWithRespectToSelf.getOfShape(this.getShape());

        long[] newPartialShape = TensorShape.concat(
            ofShape,
            wrtShapeWithoutRankLoss
        );

        DoubleTensor partialDueToSummationShapeChange = derivativeOfOutputWithRespectToSelf.get().reshape(newPartialShape);

        long[] resultShape = TensorShape.concat(
            ofShape,
            inputVertex.getShape()
        );

        DoubleTensor broadcastedPartial = DoubleTensor
            .zeros(resultShape)
            .plus(partialDueToSummationShapeChange);

        return singletonMap(inputVertex, new PartialDerivative(broadcastedPartial));
    }

    private static long[] summedOverShapeWithoutRankLoss(long[] shape, int[] sumOverDimensions) {
        long[] shapeCopy = Arrays.copyOf(shape, shape.length);
        for (int sumOverDimension : sumOverDimensions) {
            shapeCopy[sumOverDimension] = 1;
        }
        return shapeCopy;
    }

    @SaveVertexParam(DIMENSIONS_NAME)
    public int[] getOverDimensions() {
        return overDimensions;
    }
}

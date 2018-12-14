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
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.HashMap;
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
        this(inputVertex, TensorShape.dimensionRange(0, inputVertex.getShape().length));
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
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        PartialDerivatives derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInputs.get(inputVertex);
        DoubleTensor sumResult = this.getValue();
        int operandRank = inputVertex.getValue().getRank();
        return derivativeOfParentWithRespectToInputs.sumOverOfDimensions(overDimensions, sumResult.getShape(), operandRank);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {

        long[] wrtShapeWithoutRankLoss = summedOverShapeWithoutRankLoss(inputVertex.getShape(), overDimensions);

        PartialDerivatives partialDueToSummationShapeChange = getPartialDueToSummationShapeChange(derivativeOfOutputsWithRespectToSelf);

        PartialDerivatives derivativesWrtInput = partialDueToSummationShapeChange
            .multiplyAlongWrtDimensions(DoubleTensor.ones(inputVertex.getShape()), wrtShapeWithoutRankLoss);

        return singletonMap(inputVertex, derivativesWrtInput);
    }

    private PartialDerivatives getPartialDueToSummationShapeChange(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {

        long[] wrtShapeWithoutRankLoss = summedOverShapeWithoutRankLoss(inputVertex.getShape(), overDimensions);
        PartialDerivatives reshapedDiffWrtSelf = new PartialDerivatives(new HashMap<>());

        DoubleTensor partial = derivativeOfOutputsWithRespectToSelf.getValue();

        long[] newPartialShape = TensorShape.concat(
            TensorShape.selectDimensions(0, partial.getRank() - getShape().length, partial.getShape()),
            wrtShapeWithoutRankLoss
        );

        DoubleTensor reshapedPartialDerivative = derivativeOfOutputsWithRespectToSelf.getValue().reshape(newPartialShape);

        reshapedDiffWrtSelf.putWithRespectTo(derivativeOfOutputsWithRespectToSelf.getKey(), reshapedPartialDerivative);

        return reshapedDiffWrtSelf;
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

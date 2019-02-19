package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShape.getPermutedResultShapeShape;
import static io.improbable.keanu.tensor.TensorShape.invertedPermute;

public class PermuteVertex extends DoubleUnaryOpVertex implements Differentiable {

    private final static String REARRANGE_NAME = "rearrange";

    private final int[] rearrange;
    private final int[] invertedRearrange;

    @ExportVertexToPythonBindings
    public PermuteVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex,
                         @LoadVertexParam(REARRANGE_NAME) int... rearrange) {
        super(getPermutedResultShapeShape(inputVertex.getShape(), rearrange), inputVertex);
        this.rearrange = rearrange;
        this.invertedRearrange = invertedPermute(rearrange);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.permute(rearrange);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);
        int[] permuteToApply = forwardPermute(derivativeOfParentWithRespectToInputs);
        DoubleTensor result = derivativeOfParentWithRespectToInputs.get().permute(permuteToApply);
        return new PartialDerivative(result);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        int[] permuteToApply = reversePermute(derivativeOfOutputWithRespectToSelf);
        DoubleTensor result = derivativeOfOutputWithRespectToSelf.get().permute(permuteToApply);
        partials.put(inputVertex, new PartialDerivative(result));
        return partials;
    }

    private int[] forwardPermute(PartialDerivative partial) {
        int[] permuteToApply = new int[partial.get().getRank()];

        for (int i = 0; i < partial.get().getRank(); i++) {
            if (i < rearrange.length) {
                permuteToApply[i] = rearrange[i];
            } else {
                permuteToApply[i] = i;
            }
        }

        return permuteToApply;
    }

    private int[] reversePermute(PartialDerivative partial) {

        int partialRank = partial.get().getRank();
        int[] permuteToApply = new int[partialRank];
        int ofRank = partialRank - this.getRank();

        for (int i = 0; i < partialRank; i++) {
            if (i >= ofRank) {
                permuteToApply[i] = invertedRearrange[i - ofRank] + ofRank;
            } else {
                permuteToApply[i] = i;
            }
        }

        return permuteToApply;
    }

    @SaveVertexParam(REARRANGE_NAME)
    public int[] getRearrange() {
        return rearrange;
    }
}

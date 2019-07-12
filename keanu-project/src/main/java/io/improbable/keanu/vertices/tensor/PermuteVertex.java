package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShape.getPermutedIndices;
import static io.improbable.keanu.tensor.TensorShape.invertedPermute;

public class PermuteVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String REARRANGE = "rearrange";

    private final int[] rearrange;
    private final int[] invertedRearrange;

    @ExportVertexToPythonBindings
    public PermuteVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                         @LoadVertexParam(REARRANGE) int... rearrange) {
        super(getPermutedIndices(inputVertex.getShape(), rearrange), inputVertex, inputVertex.ofType());
        this.rearrange = rearrange;
        this.invertedRearrange = invertedPermute(rearrange);
    }

    @Override
    protected TENSOR op(TENSOR value) {
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

    @Override
    public boolean isDifferentiable() {
        return inputVertex.isDifferentiable();
    }

    @SaveVertexParam(REARRANGE)
    public int[] getRearrange() {
        return this.rearrange;
    }
}

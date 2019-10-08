package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
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
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        ForwardModePartialDerivative partial = derivativeOfParentsWithRespectToInput.get(inputVertex);
        int[] permuteToApply = forwardPermute(partial, partial.getWrtShape().length);
        DoubleTensor result = partial.get().permute(permuteToApply);
        return new ForwardModePartialDerivative(partial.getWrtShape(), result);
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, ReverseModePartialDerivative> partials = new HashMap<>();
        int[] permuteToApply = reversePermute(derivativeOfOutputWithRespectToSelf);
        DoubleTensor result = derivativeOfOutputWithRespectToSelf.get().permute(permuteToApply);
        partials.put(inputVertex, new ReverseModePartialDerivative(derivativeOfOutputWithRespectToSelf.getOfShape(), result));
        return partials;
    }

    private int[] forwardPermute(ForwardModePartialDerivative partial, int wrtRank) {
        int partialRank = partial.get().getRank();
        int[] permuteToApply = new int[partialRank];

        for (int i = 0; i < partialRank; i++) {
            if (i >= wrtRank) {
                permuteToApply[i] = rearrange[i - wrtRank] + wrtRank;
            } else {
                permuteToApply[i] = i;
            }
        }

        return permuteToApply;
    }

    private int[] reversePermute(ReverseModePartialDerivative partial) {

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

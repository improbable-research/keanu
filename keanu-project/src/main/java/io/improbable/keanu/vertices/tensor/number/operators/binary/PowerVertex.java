package io.improbable.keanu.vertices.tensor.number.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.BinaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.AutoDiffBroadcast;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class PowerVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String BASE_NAME = LEFT_NAME;
    private static final String EXPONENT_NAME = RIGHT_NAME;

    /**
     * Raises a vertex to the power of another
     *
     * @param base     the base vertex
     * @param exponent the exponent vertex
     */
    @ExportVertexToPythonBindings
    public PowerVertex(@LoadVertexParam(BASE_NAME) TensorVertex<T, TENSOR, VERTEX> base,
                       @LoadVertexParam(EXPONENT_NAME) TensorVertex<T, TENSOR, VERTEX> exponent) {
        super(base, exponent, base.ofType());
    }

    public TensorVertex<T, TENSOR, VERTEX> getBase() {
        return super.getLeft();
    }

    public TensorVertex<T, TENSOR, VERTEX> getExponent() {
        return super.getRight();
    }

    @Override
    protected TENSOR op(TENSOR base, TENSOR exponent) {
        return base.pow(exponent);
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        ForwardModePartialDerivative dBaseWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(left, ForwardModePartialDerivative.EMPTY);
        ForwardModePartialDerivative dExponentWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(right, ForwardModePartialDerivative.EMPTY);

        ForwardModePartialDerivative fromBase = AutoDiffBroadcast.correctForBroadcastPartialForward(dBaseWrtInput, left.getShape(), this.getShape());
        ForwardModePartialDerivative fromExponent = AutoDiffBroadcast.correctForBroadcastPartialForward(dExponentWrtInput, right.getShape(), this.getShape());

        // dc = (A ^ B) * B * (dA / A) + (dB * log (A))
        ForwardModePartialDerivative partialsFromBase;
        ForwardModePartialDerivative partialsFromExponent;

        final DoubleTensor baseValue = left.getValue().toDouble();
        final DoubleTensor exponentValue = right.getValue().toDouble();

        if (fromBase.isPresent()) {
            partialsFromBase = fromBase.multiply(
                exponentValue.times(baseValue.toDouble().pow(exponentValue.toDouble().minus(1)))
            );
        } else {
            partialsFromBase = ForwardModePartialDerivative.EMPTY;
        }

        if (fromExponent.isPresent()) {
            partialsFromExponent = fromExponent.multiply(
                baseValue.log().timesInPlace(this.getValue().toDouble())
            );
        } else {
            partialsFromExponent = ForwardModePartialDerivative.EMPTY;
        }

        return partialsFromBase.add(partialsFromExponent);
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, ReverseModePartialDerivative> partials = new HashMap<>();
        DoubleTensor baseValue = getBase().getValue().toDouble();
        DoubleTensor exponentValue = getExponent().getValue().toDouble();
        DoubleTensor basePowExponent = getValue().toDouble();
        DoubleTensor dSelfWrtBase = exponentValue.times(baseValue.pow(exponentValue.minus(1)));
        DoubleTensor dSelfWrtExponent = basePowExponent.times(baseValue.log());

        ReverseModePartialDerivative dOutputsWrtBase = derivativeOfOutputWithRespectToSelf.multiply(dSelfWrtBase);
        ReverseModePartialDerivative dOutputsWrtExponent = derivativeOfOutputWithRespectToSelf.multiply(dSelfWrtExponent);

        ReverseModePartialDerivative toBase = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtBase, this.getShape(), getBase().getShape());
        ReverseModePartialDerivative toExponent = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtExponent, this.getShape(), getExponent().getShape());

        partials.put(getBase(), toBase);
        partials.put(getExponent(), toExponent);
        return partials;
    }
}

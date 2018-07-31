package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.ParameterName.A;
import static io.improbable.keanu.distributions.dual.ParameterName.B;
import static io.improbable.keanu.distributions.dual.ParameterName.X;
import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class InverseGammaVertex extends DistributionBackedDoubleVertex<DoubleTensor> {

    /**
     * One alpha or beta or both driving an arbitrarily shaped tensor of Inverse Gamma
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param alpha       the alpha of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     * @param beta        the beta of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
     */
    //package private
    InverseGammaVertex(int[] tensorShape, DoubleVertex alpha, DoubleVertex beta) {
        super(tensorShape, DistributionOfType::inverseGamma, alpha, beta);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        ParameterMap<DoubleTensor> dlnP = distribution().dLogProb(value);

        return convertDualNumbersToDiff(dlnP.get(A).getValue(), dlnP.get(B).getValue(), dlnP.get(X).getValue());
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdalpha,
                                                             DoubleTensor dLogPdbeta,
                                                             DoubleTensor dLogPdx) {

        Differentiator differentiator = new Differentiator();
        PartialDerivatives dLogPdInputsFromA = differentiator.calculateDual(getAlpha()).getPartialDerivatives().multiplyBy(dLogPdalpha);
        PartialDerivatives dLogPdInputsFromB = differentiator.calculateDual(getBeta()).getPartialDerivatives().multiplyBy(dLogPdbeta);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromA.add(dLogPdInputsFromB);

        if (!this.isObserved()) {
            dLogPdInputs.putWithRespectTo(getId(), dLogPdx.reshape(
                shapeToDesiredRankByPrependingOnes(dLogPdx.getShape(), dLogPdx.getRank() + getValue().getRank()))
            );
        }

        PartialDerivatives summed = dLogPdInputs.sum(true, TensorShape.dimensionRange(0, getShape().length));
        return summed.asMap();
    }

    private DoubleVertex getAlpha() {
        return (DoubleVertex) getParents().get(0);
    }

    private DoubleVertex getBeta() {
        return (DoubleVertex) getParents().get(1);
    }
}

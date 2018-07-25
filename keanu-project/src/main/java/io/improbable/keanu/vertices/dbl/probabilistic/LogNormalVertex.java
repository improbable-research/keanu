package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.ParameterName.MU;
import static io.improbable.keanu.distributions.dual.ParameterName.SIGMA;
import static io.improbable.keanu.distributions.dual.ParameterName.X;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;

public class LogNormalVertex extends DistributionBackedDoubleVertex<DoubleTensor> {


    /**
     * One mu or s or both driving an arbitrarily shaped tensor of LogNormal
     * https://en.wikipedia.org/wiki/Log-normal_distribution
     *
     * @param tensorShape the desired shape of the vertex
     * @param mu          the mu (location) of the LogNormal with either the same tensor shape as specified for this
     *                    vertex or mu scalar
     * @param sigma       the sigma of the Logistic with either the same shape as specified for this vertex or mu scalar
     */
    // package private
    LogNormalVertex(int[] tensorShape, DoubleVertex mu, DoubleVertex sigma) {
        super(tensorShape, DistributionOfType::logNormal, mu, sigma);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        ParameterMap<DoubleTensor> dlnP = distribution().dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(MU).getValue(), dlnP.get(SIGMA).getValue(), dlnP.get(X).getValue());
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdmu,
                                                             DoubleTensor dLogPdsigma,
                                                             DoubleTensor dLogPdx) {
        Differentiator differentiator = new Differentiator();
        PartialDerivatives dLogPdInputsFromMu = differentiator.calculateDual((Differentiable) getMu()).getPartialDerivatives().multiplyBy(dLogPdmu);
        PartialDerivatives dLogPdInputsFromSigma = differentiator.calculateDual((Differentiable) getSigma()).getPartialDerivatives().multiplyBy(dLogPdsigma);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromMu.add(dLogPdInputsFromSigma);

        if (!this.isObserved()) {
            dLogPdInputs.putWithRespectTo(getId(), dLogPdx.reshape(
                shapeToDesiredRankByPrependingOnes(dLogPdx.getShape(), dLogPdx.getRank() + getValue().getRank()))
            );
        }

        PartialDerivatives summed = dLogPdInputs.sum(true, TensorShape.dimensionRange(0, getShape().length));
        return summed.asMap();
    }

    private DoubleVertex getMu() {
        return (DoubleVertex) getParents().get(0);
    }

    private DoubleVertex getSigma() {
        return (DoubleVertex) getParents().get(1);
    }
}

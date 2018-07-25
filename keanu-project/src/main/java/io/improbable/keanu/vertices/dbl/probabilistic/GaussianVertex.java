package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.ParameterName.MU;
import static io.improbable.keanu.distributions.dual.ParameterName.SIGMA;
import static io.improbable.keanu.distributions.dual.ParameterName.X;
import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class GaussianVertex extends DistributionBackedDoubleVertex<DoubleTensor> {


    /**
     * One mu or sigma or both that match a proposed tensor shape of Gaussian
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param mu          the mu of the Gaussian with either the same tensorShape as specified for this vertex or a scalar
     * @param sigma       the sigma of the Gaussian with either the same tensorShape as specified for this vertex or a scalar
     */
    // package private
    GaussianVertex(int[] tensorShape, DoubleVertex mu, DoubleVertex sigma) {
        super(tensorShape, DistributionOfType::gaussian, mu, sigma);
    }

    public Vertex<?> getMu() {
        return getParents().get(0);
    }

    public Vertex<?> getSigma() {
        return getParents().get(1);
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


}

package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.ParameterName.BETA;
import static io.improbable.keanu.distributions.dual.ParameterName.MU;
import static io.improbable.keanu.distributions.dual.ParameterName.X;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class LaplaceVertex extends DistributionBackedDoubleVertex<DoubleTensor> {

    /**
     * One mu or beta or both that match a proposed tensor shape of Laplace
     *
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor within the vertex
     * @param mu          the mu of the Laplace with either the same shape as specified for this vertex or a scalar
     * @param beta        the beta of the Laplace with either the same shape as specified for this vertex or a scalar
     */
    // package private
    LaplaceVertex(int[] tensorShape, DoubleVertex mu, DoubleVertex beta) {
        super(tensorShape, DistributionOfType::laplace, mu, beta);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        ParameterMap<DoubleTensor> dlnP = distribution().dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(MU).getValue(), dlnP.get(BETA).getValue(), dlnP.get(X).getValue());
    }

    private DoubleVertex getMu() {
        return (DoubleVertex) getParents().get(0);
    }

    private DoubleVertex getBeta() {
        return (DoubleVertex) getParents().get(1);
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdmu,
                                                             DoubleTensor dPdbeta,
                                                             DoubleTensor dPdx) {

        Differentiator differentiator = new Differentiator();
        PartialDerivatives dPdInputsFromMu = differentiator.calculateDual((Differentiable)getMu()).getPartialDerivatives().multiplyBy(dPdmu);
        PartialDerivatives dPdInputsFromBeta = differentiator.calculateDual((Differentiable)getBeta()).getPartialDerivatives().multiplyBy(dPdbeta);
        PartialDerivatives dPdInputs = dPdInputsFromMu.add(dPdInputsFromBeta);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }
}

package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class BetaVertex extends DistributionBackedDoubleVertex<DoubleTensor> {

    private static final Function<List<DoubleTensor>,ContinuousDistribution> createBetaDistribution = (l) -> {
        return DistributionOfType.beta(l.get(0), l.get(1), DoubleTensor.scalar(0.), DoubleTensor.scalar(1.));
    };

    /**
     * One alpha or beta or both that match a proposed tensor shape of Beta.
     *
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor contained in the vertex
     * @param alpha       the alpha of the Beta with either the same tensorShape as specified for this vertex or a scalar
     * @param beta        the beta of the Beta with either the same tensorShape as specified for this vertex or a scalar
     */
    // package private
    BetaVertex(int[] tensorShape, DoubleVertex alpha, DoubleVertex beta) {
        super(tensorShape, createBetaDistribution, alpha, beta);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        ParameterMap<DoubleTensor> dlnP = distribution().dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(ParameterName.A).getValue(), dlnP.get(ParameterName.B).getValue(), dlnP.get(ParameterName.X).getValue());
    }

    private Map<Long,DoubleTensor> convertDualNumbersToDiff(DoubleTensor dPdalpha, DoubleTensor dPdbeta, DoubleTensor dPdx) {
            Differentiator differentiator = new Differentiator();
            PartialDerivatives dPdInputsFromAlpha = differentiator.calculateDual((Differentiable) getAlpha()).getPartialDerivatives().multiplyBy(dPdalpha);
            PartialDerivatives dPdInputsFromBeta = differentiator.calculateDual((Differentiable) getBeta()).getPartialDerivatives().multiplyBy(dPdbeta);
        PartialDerivatives dPdInputs = dPdInputsFromAlpha.add(dPdInputsFromBeta);

        if (!this.isObserved()) {
            dPdInputs.putWithRespectTo(getId(), dPdx);
        }

        return dPdInputs.asMap();
    }

    private Vertex<?> getAlpha() {
        return getParents().get(0);
    }

    private Vertex<?> getBeta() {
        return getParents().get(1);
    }
}

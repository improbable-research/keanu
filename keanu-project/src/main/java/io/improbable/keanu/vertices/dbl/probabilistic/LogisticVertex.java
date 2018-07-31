package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.ParameterName.MU;
import static io.improbable.keanu.distributions.dual.ParameterName.S;
import static io.improbable.keanu.distributions.dual.ParameterName.X;
import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class LogisticVertex extends DistributionBackedDoubleVertex<DoubleTensor> {

    /**
     * One mu or s or both driving an arbitrarily shaped tensor of Logistic
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param mu          the mu (location) of the Logistic with either the same shape as specified for this vertex or mu scalar
     * @param s           the s (scale) of the Logistic with either the same shape as specified for this vertex or mu scalar
     */
    // package private
    LogisticVertex(int[] tensorShape, DoubleVertex mu, DoubleVertex s) {
    super(tensorShape, DistributionOfType::logistic, mu, s);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        ParameterMap<DoubleTensor> dlnP = distribution().dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(MU).getValue(), dlnP.get(S).getValue(), dlnP.get(X).getValue());
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdmu,
                                                             DoubleTensor dLogPds,
                                                             DoubleTensor dLogPdx) {

        Differentiator differentiator = new Differentiator();
        PartialDerivatives dLogPdInputsFromA = differentiator.calculateDual(getMu()).getPartialDerivatives().multiplyBy(dLogPdmu);
        PartialDerivatives dLogPdInputsFromB = differentiator.calculateDual(getS()).getPartialDerivatives().multiplyBy(dLogPds);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromA.add(dLogPdInputsFromB);


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

    private DoubleVertex getS() {
        return (DoubleVertex) getParents().get(1);
    }
}

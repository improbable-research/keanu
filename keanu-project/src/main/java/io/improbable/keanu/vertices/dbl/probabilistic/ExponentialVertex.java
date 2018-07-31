package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.ParameterName.LAMBDA;
import static io.improbable.keanu.distributions.dual.ParameterName.LOCATION;
import static io.improbable.keanu.distributions.dual.ParameterName.X;
import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ExponentialVertex extends DistributionBackedDoubleVertex<DoubleTensor> {


    /**
     * One location or lambda or both driving an arbitrarily shaped tensor of Exponential
     * <p>
     * pdf = lambda * exp(-lambda*x)
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param location    the horizontal shift of the Exponential with either the same shape as specified
     *                    for this vertex or location scalar
     * @param lambda      the lambda of the Exponential with either the same shape as specified for this
     *                    vertex or location scalar.
     */
    // package private
    ExponentialVertex(int[] tensorShape, DoubleVertex location, DoubleVertex lambda) {
        super(tensorShape, DistributionOfType::exponential, location, lambda);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        ParameterMap<DoubleTensor> dlnP = distribution().dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(LOCATION).getValue(), dlnP.get(LAMBDA).getValue(), dlnP.get(X).getValue());
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdlocation,
                                                             DoubleTensor dLogPdlambda,
                                                             DoubleTensor dLogPdx) {

        Differentiator differentiator = new Differentiator();
        PartialDerivatives dLogPdInputsFromA = differentiator.calculateDual(getLocation()).getPartialDerivatives().multiplyBy(dLogPdlocation);
        PartialDerivatives dLogPdInputsFromB = differentiator.calculateDual(getLambda()).getPartialDerivatives().multiplyBy(dLogPdlambda);
        PartialDerivatives dLogPdInputs = dLogPdInputsFromA.add(dLogPdInputsFromB);

        if (!this.isObserved()) {
            dLogPdInputs.putWithRespectTo(getId(), dLogPdx.reshape(
                shapeToDesiredRankByPrependingOnes(dLogPdx.getShape(), dLogPdx.getRank() + getValue().getRank()))
            );
        }

        PartialDerivatives summed = dLogPdInputs.sum(true, TensorShape.dimensionRange(0, getShape().length));
        return summed.asMap();
    }

    private DoubleVertex getLocation() {
        return (DoubleVertex) getParents().get(0);
    }

    private DoubleVertex getLambda() {
        return (DoubleVertex) getParents().get(1);
    }
}



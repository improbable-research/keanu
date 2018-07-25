package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class ChiSquaredVertex extends DistributionBackedDoubleVertex<IntegerTensor> {

    /**
     * One k that must match a proposed tensor shape of ChiSquared
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param k           the number of degrees of freedom
     */
    // package private
    ChiSquaredVertex(int[] tensorShape, IntegerVertex k) {
        super(tensorShape, DistributionOfType::chiSquared, k);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        throw new UnsupportedOperationException();
    }

}

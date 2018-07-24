package io.improbable.keanu.vertices.intgr.probabilistic;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class PoissonVertex extends DistributionBackedIntegerVertex<DoubleTensor> {

    /**
     * One mu that must match a proposed tensor shape of Poisson.
     *
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param mu    the mu of the Poisson with either the same shape as specified for this vertex or a scalar
     */
    // TODO: make package private
    public PoissonVertex(int[] tensorShape, DoubleVertex mu) {
        super(tensorShape, DistributionOfType::poisson, mu);
    }

    public Vertex<?> getMu() {
        return getParents().get(0);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(IntegerTensor value) {
        throw new UnsupportedOperationException();
    }
}

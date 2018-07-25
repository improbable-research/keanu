package io.improbable.keanu.vertices.intgr.probabilistic;

import java.util.List;
import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class UniformIntVertex extends DistributionBackedIntegerVertex<IntegerTensor> {

    /**
     * @param shape tensor shape of value
     * @param min   The inclusive lower bound.
     * @param max   The exclusive upper bound.
     */
    public UniformIntVertex(int[] shape, IntegerVertex min, IntegerVertex max) {
        super(shape, DistributionOfType::uniformInt, min, max);
    }

    @Override
    public List<IntegerVertex> getParents() {
        return (List<IntegerVertex>) super.getParents();
    }

    public Vertex<IntegerTensor> getMin() {
        return getParents().get(0);
    }

    public Vertex<IntegerTensor> getMax() {
        return getParents().get(1);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(IntegerTensor value) {
        throw new UnsupportedOperationException();
    }
}

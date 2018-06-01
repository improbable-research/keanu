package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Map;

public class PoissonVertex extends ProbabilisticInteger {

    private final DoubleVertex mu;

    public PoissonVertex(DoubleVertex mu) {
        this.mu = mu;
        setParents(mu);
    }

    public PoissonVertex(double mu) {
        this(new ConstantDoubleVertex(mu));
    }

    public Vertex<Double> getMu() {
        return mu;
    }

    @Override
    public double logPmf(Integer value) {
        return Math.log(Poisson.pmf(mu.getValue(), value));
    }

    @Override
    public Map<Long, DoubleTensor> dLogPmf(Integer value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Integer sample(KeanuRandom random) {
        return Poisson.sample(mu.getValue(), random);
    }
}

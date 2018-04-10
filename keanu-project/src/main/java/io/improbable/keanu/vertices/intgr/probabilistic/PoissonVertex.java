package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Map;
import java.util.Random;

public class PoissonVertex extends ProbabilisticInteger {

    private final Random random;
    private final Vertex<Double> mu;

    public PoissonVertex(Vertex<Double> mu, Random random) {
        this.mu = mu;
        this.random = random;
        setParents(mu);
        setValue(sample());
    }

    public PoissonVertex(double mu, Random random) {
        this(new ConstantDoubleVertex(mu), random);
    }

    @Override
    public double density(Integer value) {
        return Poisson.pdf(mu.getValue(), value);
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Integer sample() {
        return new Poisson(mu.getValue(), random).sample();
    }
}
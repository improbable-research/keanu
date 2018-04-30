package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Map;
import java.util.Random;

public class PoissonVertex extends ProbabilisticInteger {

    private final Random random;
    private final DoubleVertex mu;

    public PoissonVertex(DoubleVertex mu, Random random) {
        this.mu = mu;
        this.random = random;
        setParents(mu);
    }

    public PoissonVertex(double mu, Random random) {
        this(new ConstantDoubleVertex(mu), random);
    }

    public PoissonVertex(DoubleVertex mu) {
        this(mu, new Random());
    }

    public PoissonVertex(double mu) {
        this(new ConstantDoubleVertex(mu), new Random());
    }

    public Vertex<Double> getMu() {
        return mu;
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

package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Map;

public class LaplaceDistribution extends ProbabilisticDouble {

    private final DoubleVertex mu;
    private final DoubleVertex beta;

    public LaplaceDistribution(DoubleVertex mu, DoubleVertex beta) {
        this.mu = mu;
        this.beta = beta;
        setValue(sample());
        setParents(mu, beta);
    }

    public LaplaceDistribution(double mu, double beta) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(beta));
    }

    @Override
    public double density(Double value) {
        return 0;
    }

    public double logDensity(Double value) {
        return 0;
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        return null;
    }

    @Override
    public Map<String, Double> dlnDensityAtValue() {
        return null;
    }

    @Override
    public Double sample() {
        return null;
    }
}

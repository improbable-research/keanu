package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Chi2;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.Map;
import java.util.Random;

public class Chi2Vertex extends ProbabilisticDouble {

    private IntegerVertex k;
    private Random random;

    public Chi2Vertex(IntegerVertex k, Random random) {
        this.k = k;
        this.random = random;
        setValue(sample());
        setParents(k);
    }

    public Chi2Vertex(int k) {
        this(new ConstantIntegerVertex(k), new Random());
    }

    @Override
    public Double sample() {
        return Chi2.sample(k.getValue(), random);
    }

    @Override
    public double density(Double value) {
        return Chi2.pdf(k.getValue(), value);
    }

    public double logDensity(Double value) {
        return Chi2.logPdf(k.getValue(), value);
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        throw new UnsupportedOperationException();
    }

}

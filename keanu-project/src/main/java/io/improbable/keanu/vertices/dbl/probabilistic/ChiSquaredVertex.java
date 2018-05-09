package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.ChiSquared;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.Map;
import java.util.Random;

public class ChiSquaredVertex extends ProbabilisticDouble {

    private IntegerVertex k;
    private Random random;

    public ChiSquaredVertex(IntegerVertex k, Random random) {
        this.k = k;
        this.random = random;
        setParents(k);
    }

    public ChiSquaredVertex(IntegerVertex k) {
        this(k, new Random());
    }

    public ChiSquaredVertex(int k, Random random) {
        this(new ConstantIntegerVertex(k), random);
    }

    public ChiSquaredVertex(int k) {
        this(new ConstantIntegerVertex(k), new Random());
    }

    @Override
    public Double sample() {
        return ChiSquared.sample(k.getValue(), random);
    }

    @Override
    public double logPdf(Double value) {
        return ChiSquared.logPdf(k.getValue(), value);
    }

    @Override
    public Map<String, DoubleTensor> dLogPdf(Double value) {
        throw new UnsupportedOperationException();
    }

}

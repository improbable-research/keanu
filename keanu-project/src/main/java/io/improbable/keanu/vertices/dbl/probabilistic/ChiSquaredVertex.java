package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.ChiSquared;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.util.Map;
import java.util.Random;

public class ChiSquaredVertex extends ProbabilisticDouble {

    private IntegerVertex k;

    public ChiSquaredVertex(IntegerVertex k) {
        this.k = k;
        setParents(k);
    }

    public ChiSquaredVertex(int k) {
        this(new ConstantIntegerVertex(k));
    }

    @Override
    public Double sample(Random random) {
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

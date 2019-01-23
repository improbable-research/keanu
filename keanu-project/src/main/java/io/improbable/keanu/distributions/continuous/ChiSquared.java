package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ChiSquared implements ContinuousDistribution {

    private static final double LOG_TWO = Math.log(2);
    private final IntegerTensor k;

    public static ContinuousDistribution withParameters(IntegerTensor k) {
        return new ChiSquared(k);
    }

    private ChiSquared(IntegerTensor k) {
        this.k = k;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        return random.nextGamma(shape, DoubleTensor.TWO_SCALAR, k.toDouble().div(2));
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor halfK = k.toDouble().div(2);
        final DoubleTensor numerator = halfK.minus(1).timesInPlace(x.log()).minusInPlace(x.div(2));
        final DoubleTensor denominator = halfK.times(LOG_TWO).plusInPlace(halfK.logGamma());
        return numerator.minusInPlace(denominator);
    }

    public static DoubleVertex logProbOutput(LogProbGraph.DoublePlaceholderVertex x, LogProbGraph.IntegerPlaceHolderVertex k) {
        final DoubleVertex halfK = k.toDouble().div(2.);
        final DoubleVertex numerator = halfK.minus(1.).times(x.log()).minus(x.div(2.));
        final DoubleVertex denominator = halfK.times(LOG_TWO).plus(halfK.logGamma());
        return numerator.minus(denominator);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        throw new UnsupportedOperationException();
    }
}
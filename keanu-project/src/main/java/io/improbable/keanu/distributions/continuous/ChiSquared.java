package io.improbable.keanu.distributions.continuous;

import org.apache.commons.math3.special.Gamma;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ChiSquared implements ContinuousDistribution {

    private static final double LOG_TWO = Math.log(2);
    private final IntegerTensor k;

    // package private
    ChiSquared(IntegerTensor k) {
        this.k = k;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        return random.nextGamma(shape, DoubleTensor.ZERO_SCALAR, DoubleTensor.TWO_SCALAR, k.toDouble().div(2));
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        DoubleTensor halfK = k.toDouble().div(2);
        DoubleTensor numerator = halfK.minus(1).timesInPlace(x.log()).minusInPlace(x.div(2));
        DoubleTensor denominator = halfK.times(LOG_TWO).plusInPlace(halfK.apply(Gamma::gamma).logInPlace());
        return numerator.minusInPlace(denominator);
    }

    @Override
    public ParameterMap<DoubleTensor> dLogProb(DoubleTensor x) {
        throw new UnsupportedOperationException();
    }
}
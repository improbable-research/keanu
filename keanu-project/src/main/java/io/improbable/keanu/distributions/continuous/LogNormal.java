package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.continuous.Gaussian.LN_SQRT_2PI;
import static io.improbable.keanu.distributions.dual.ParameterName.MU;
import static io.improbable.keanu.distributions.dual.ParameterName.SIGMA;
import static io.improbable.keanu.distributions.dual.ParameterName.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class LogNormal implements ContinuousDistribution {

    private final DoubleTensor mu;
    private final DoubleTensor sigma;

    /**
     * @param mu     location parameter (any real number)
     * @param sigma  square root of variance (greater than 0)
     * @return       a new ContinuousDistribution object
     */
    // package private
    LogNormal(DoubleTensor mu, DoubleTensor sigma) {
        this.mu = mu;
        this.sigma = sigma;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        return DistributionOfType.gaussian(mu, sigma).sample(shape, random).expInPlace();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor lnSigmaX = sigma.times(x).logInPlace();
        final DoubleTensor lnXMinusMuSquared = x.log().minusInPlace(mu).powInPlace(2);
        final DoubleTensor lnXMinusMuSquaredOver2Variance = lnXMinusMuSquared.divInPlace(sigma.pow(2).timesInPlace(2.0));
        return lnXMinusMuSquaredOver2Variance.plusInPlace(lnSigmaX).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    @Override
    public ParameterMap<DoubleTensor> dLogProb(DoubleTensor x) {
        final DoubleTensor variance = sigma.pow(2);
        final DoubleTensor lnXMinusMu = x.log().minusInPlace(mu);

        final DoubleTensor dLogPdmu = lnXMinusMu.div(variance);
        final DoubleTensor dLogPdx = dLogPdmu.plus(1.0).unaryMinus().divInPlace(x);
        final DoubleTensor dLogPdsigma = lnXMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(sigma))
            .minusInPlace(sigma.reciprocal());

        return new ParameterMap<DoubleTensor>()
            .put(MU, dLogPdmu)
            .put(SIGMA, dLogPdsigma)
            .put(X, dLogPdx);
    }
}

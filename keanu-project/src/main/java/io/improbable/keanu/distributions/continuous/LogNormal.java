package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.continuous.Gaussian.LN_SQRT_2PI;
import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static io.improbable.keanu.distributions.dual.Diffs.SIGMA;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
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
    public static ContinuousDistribution withParameters(DoubleTensor mu, DoubleTensor sigma) {
        return new LogNormal(mu, sigma);
    }

    private LogNormal(DoubleTensor mu, DoubleTensor sigma) {
        this.mu = mu;
        this.sigma = sigma;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        return Gaussian.withParameters(mu, sigma).sample(shape, random).expInPlace();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor lnSigmaX = sigma.times(x).logInPlace();
        final DoubleTensor lnXMinusMuSquared = x.log().minusInPlace(mu).powInPlace(2);
        final DoubleTensor lnXMinusMuSquaredOver2Variance = lnXMinusMuSquared.divInPlace(sigma.pow(2).timesInPlace(2.0));
        return lnXMinusMuSquaredOver2Variance.plusInPlace(lnSigmaX).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor variance = sigma.pow(2);
        final DoubleTensor lnXMinusMu = x.log().minusInPlace(mu);

        final DoubleTensor dLogPdmu = lnXMinusMu.div(variance);
        final DoubleTensor dLogPdx = dLogPdmu.plus(1.0).unaryMinus().divInPlace(x);
        final DoubleTensor dLogPdsigma = lnXMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(sigma))
            .minusInPlace(sigma.reciprocal());

        return new Diffs()
            .put(MU, dLogPdmu)
            .put(SIGMA, dLogPdsigma)
            .put(X, dLogPdx);
    }
}

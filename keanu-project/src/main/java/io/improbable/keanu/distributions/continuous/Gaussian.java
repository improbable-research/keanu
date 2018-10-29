package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.distributions.hyperparam.Diffs.MU;
import static io.improbable.keanu.distributions.hyperparam.Diffs.SIGMA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

public class Gaussian implements ContinuousDistribution {

    public static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    public static final double LN_SQRT_2PI = Math.log(SQRT_2PI);
    private final DoubleTensor mu;
    private final DoubleTensor sigma;

    public static ContinuousDistribution withParameters(DoubleTensor mu, DoubleTensor sigma) {
        return new Gaussian(mu, sigma);
    }

    private Gaussian(DoubleTensor mu, DoubleTensor sigma) {
        this.mu = mu;
        this.sigma = sigma;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        DoubleTensor unityGaussian = random.nextGaussian(shape);
        return unityGaussian.timesInPlace(sigma).plusInPlace(mu);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor lnSigma = sigma.log();
        final DoubleTensor xMinusMuSquared = x.minus(mu).powInPlace(2);
        final DoubleTensor xMinusMuSquaredOver2Variance = xMinusMuSquared.divInPlace(sigma.pow(2).timesInPlace(2.0));
        return xMinusMuSquaredOver2Variance.plusInPlace(lnSigma).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor variance = sigma.pow(2);
        final DoubleTensor xMinusMu = x.minus(mu);

        final DoubleTensor dLogPdmu = xMinusMu.div(variance);
        final DoubleTensor dLogPdx = dLogPdmu.unaryMinus();
        final DoubleTensor dLogPdsigma = xMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(sigma))
            .minusInPlace(sigma.reciprocal());

        return new Diffs()
            .put(MU, dLogPdmu)
            .put(SIGMA, dLogPdsigma)
            .put(X, dLogPdx);
    }

}

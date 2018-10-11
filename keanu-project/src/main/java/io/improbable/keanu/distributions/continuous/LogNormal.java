package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.continuous.Gaussian.LN_SQRT_2PI;
import static io.improbable.keanu.distributions.hyperparam.Diffs.MU;
import static io.improbable.keanu.distributions.hyperparam.Diffs.SIGMA;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class LogNormal implements ContinuousDistribution {

    private final DoubleVertex x;
    private final DoubleVertex mu;
    private final DoubleVertex sigma;
    private final Gaussian gaussian;

    /**
     * @param x     domain value
     * @param mu    location parameter (any real number)
     * @param sigma square root of variance (greater than 0)
     * @return a new ContinuousDistribution object
     */
    public static LogNormal withParameters(DoubleVertex x, DoubleVertex mu, DoubleVertex sigma) {
        return new LogNormal(x, mu, sigma);
    }

    private LogNormal(DoubleVertex x, DoubleVertex mu, DoubleVertex sigma) {
        this.x = x;
        this.mu = mu;
        this.sigma = sigma;
        this.gaussian = Gaussian.withParameters(x, mu, sigma);
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        return gaussian.sample(shape, random).expInPlace();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor xValue) {
        final DoubleTensor muValue = mu.getValue();
        final DoubleTensor sigmaValue = sigma.getValue();

        final DoubleTensor lnSigmaX = sigmaValue.times(xValue).logInPlace();
        final DoubleTensor lnXMinusMuSquared = xValue.log().minusInPlace(muValue).powInPlace(2);
        final DoubleTensor lnXMinusMuSquaredOver2Variance = lnXMinusMuSquared.divInPlace(sigmaValue.pow(2).timesInPlace(2.0));
        return lnXMinusMuSquaredOver2Variance.plusInPlace(lnSigmaX).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    @Override
    public Diffs dLogProb(DoubleTensor xValue) {
        final DoubleTensor muValue = mu.getValue();
        final DoubleTensor sigmaValue = sigma.getValue();

        final DoubleTensor variance = sigmaValue.pow(2);
        final DoubleTensor lnXMinusMu = xValue.log().minusInPlace(muValue);

        final DoubleTensor dLogPdmu = lnXMinusMu.div(variance);
        final DoubleTensor dLogPdx = dLogPdmu.plus(1.0).unaryMinus().divInPlace(xValue);
        final DoubleTensor dLogPdsigma = lnXMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(sigmaValue))
            .minusInPlace(sigmaValue.reciprocal());

        return new Diffs()
            .put(MU, dLogPdmu)
            .put(SIGMA, dLogPdsigma)
            .put(X, dLogPdx);
    }
}

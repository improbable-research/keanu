package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

public class Gaussian implements ContinuousDistribution {

    public static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    public static final double LN_SQRT_2PI = Math.log(SQRT_2PI);
    private final DoubleTensor mu;
    private final DoubleTensor sigma;

    public static Gaussian withParameters(DoubleTensor mu, DoubleTensor sigma) {
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
        final DoubleTensor xMinusMuSquared = x.minus(mu).powInPlace(2.);
        final DoubleTensor xMinusMuSquaredOver2Variance = xMinusMuSquared.divInPlace(sigma.pow(2.).timesInPlace(2.));
        return xMinusMuSquaredOver2Variance.plusInPlace(lnSigma).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex mu, DoublePlaceholderVertex sigma) {
        final DoubleVertex lnSigma = sigma.log();
        final DoubleVertex xMinusMuSquared = x.minus(mu).pow(2.);
        final DoubleVertex xMinusMuSquaredOver2Variance = xMinusMuSquared.div(sigma.pow(2.).times(2.));
        return xMinusMuSquaredOver2Variance.plus(lnSigma).plus(LN_SQRT_2PI).unaryMinus();
    }

    public DoubleTensor[] dLogProb(DoubleTensor x, boolean wrtX, boolean wrtMu, boolean wrtSigma) {
        final DoubleTensor variance = sigma.pow(2);
        final DoubleTensor xMinusMu = x.minus(mu);

        DoubleTensor[] diff = new DoubleTensor[3];

        if (wrtX || wrtMu) {
            final DoubleTensor dLogPdMu = xMinusMu.div(variance);

            if (wrtX) {
                final DoubleTensor dLogPdx = dLogPdMu.unaryMinus();
                diff[0] = dLogPdx;
            }

            if (wrtMu) {
                diff[1] = dLogPdMu;
            }
        }

        if (wrtSigma) {
            final DoubleTensor dLogPdSigma = xMinusMu.powInPlace(2.0)
                .divInPlace(variance.timesInPlace(sigma))
                .minusInPlace(sigma.reciprocal());

            diff[2] = dLogPdSigma;
        }

        return diff;
    }

}

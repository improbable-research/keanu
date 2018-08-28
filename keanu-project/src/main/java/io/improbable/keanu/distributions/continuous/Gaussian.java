package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.Distribution;
import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static io.improbable.keanu.distributions.dual.Diffs.SIGMA;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

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
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
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

    @Override
    public DoubleTensor computeKLDivergence(Distribution<DoubleTensor> q) {
        if (q instanceof Gaussian) {
            DoubleTensor qMu = ((Gaussian) q).mu;
            DoubleTensor qSigma = ((Gaussian) q).sigma;

            DoubleTensor qSigmaDivPSigmaLog = qSigma.div(sigma).logInPlace();

            DoubleTensor pMuMinusQMuPow2 = mu.minus(qMu).powInPlace(2);
            DoubleTensor qSigmaPow2Times2 = qSigma.pow(2).timesInPlace(2);
            DoubleTensor pSigmaPow2PlusMuDiffSquaredDivQSigmaPow2Times2 = sigma.pow(2).plusInPlace(pMuMinusQMuPow2).divInPlace(qSigmaPow2Times2);

            return qSigmaDivPSigmaLog.plusInPlace(pSigmaPow2PlusMuDiffSquaredDivQSigmaPow2Times2).minusInPlace(0.5);
        } else {
            return ContinuousDistribution.super.computeKLDivergence(q);
        }
    }
}

package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static io.improbable.keanu.distributions.dual.Diffs.SIGMA;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier,
 * ARL-TR-2168 March 2000,
 * 5.1.16 page 29"
 */
public class Gaussian implements ContinuousDistribution {

    private static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    static final double LN_SQRT_2PI = Math.log(SQRT_2PI);
    private final DoubleTensor mean;
    private final DoubleTensor standardDeviation;

    /**
     * @param mean              the mean of Gaussian Distribution
     * @param standardDeviation the standard deviation of Gaussian Distribution
     * @return an instance of {@link ContinuousDistribution}
     */
    public static ContinuousDistribution withParameters(DoubleTensor mean, DoubleTensor standardDeviation) {
        return new Gaussian(mean, standardDeviation);
    }

    private Gaussian(DoubleTensor mean, DoubleTensor standardDeviation) {
        this.mean = mean;
        this.standardDeviation = standardDeviation;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor unityGaussian = random.nextGaussian(shape);
        return unityGaussian.timesInPlace(standardDeviation).plusInPlace(mean);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor lnSigma = standardDeviation.log();
        final DoubleTensor xMinusMuSquared = x.minus(mean).powInPlace(2);
        final DoubleTensor xMinusMuSquaredOver2Variance = xMinusMuSquared.divInPlace(standardDeviation.pow(2).timesInPlace(2.0));
        return xMinusMuSquaredOver2Variance.plusInPlace(lnSigma).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor variance = standardDeviation.pow(2);
        final DoubleTensor xMinusMu = x.minus(mean);

        final DoubleTensor dLogPdmu = xMinusMu.div(standardDeviation);
        final DoubleTensor dLogPdx = dLogPdmu.unaryMinus();
        final DoubleTensor dLogPdsigma = xMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(standardDeviation))
            .minusInPlace(standardDeviation.reciprocal());

        return new Diffs()
            .put(MU, dLogPdmu)
            .put(SIGMA, dLogPdsigma)
            .put(X, dLogPdx);
    }

}
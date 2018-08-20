package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.A;
import static io.improbable.keanu.distributions.dual.Diffs.B;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import org.apache.commons.math3.special.Gamma;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.2 page 14"
 */
public class Beta implements ContinuousDistribution {

    private final DoubleTensor distributionShape1;
    private final DoubleTensor distributionShape2;
    private final DoubleTensor xMin;
    private final DoubleTensor xMax;

    /**
     * @param distributionShape1 shape parameter (not to be confused with tensor shape)
     * @param distributionShape2 shape parameter (not to be confused with tensor shape)
     * @param xMin               minimum value of random variable x
     * @param xMax               maximum value of random variable x
     * @return an instance of {@link ContinuousDistribution}
     */
    public static ContinuousDistribution withParameters(DoubleTensor distributionShape1, DoubleTensor distributionShape2, DoubleTensor xMin, DoubleTensor xMax) {
        return new Beta(distributionShape1, distributionShape2, xMin, xMax);
    }

    private Beta(DoubleTensor distributionShape1, DoubleTensor distributionShape2, DoubleTensor xMin, DoubleTensor xMax) {
        this.distributionShape1 = distributionShape1;
        this.distributionShape2 = distributionShape2;
        this.xMin = xMin;
        this.xMax = xMax;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {

        final DoubleTensor y1 = random.nextGamma(shape, DoubleTensor.ONE_SCALAR, distributionShape1);
        final DoubleTensor y2 = random.nextGamma(shape, DoubleTensor.ONE_SCALAR, distributionShape2);

        final DoubleTensor range = xMax.minus(xMin);
        final DoubleTensor y1PlusY2 = y1.plus(y2);

        final DoubleTensor lessThan = xMax.minus(y2.div(y1PlusY2).timesInPlace(range));
        final DoubleTensor greaterThan = xMin.plus(y1.div(y1PlusY2).timesInPlace(range));

        final DoubleTensor lessMask = distributionShape1.getLessThanMask(distributionShape2);
        final DoubleTensor greaterMask = distributionShape1.getGreaterThanOrEqualToMask(distributionShape2);

        return lessMask.timesInPlace(lessThan).plusInPlace(greaterMask.timesInPlace(greaterThan));
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor lnGammaAlpha = distributionShape1.apply(Gamma::logGamma);
        final DoubleTensor lnGammaBeta = distributionShape2.apply(Gamma::logGamma);
        final DoubleTensor alphaPlusBetaLnGamma = (distributionShape1.plus(distributionShape2)).applyInPlace(Gamma::logGamma);
        final DoubleTensor alphaMinusOneTimesLnX = x.log().timesInPlace(distributionShape1.minus(1));
        final DoubleTensor betaMinusOneTimesOneMinusXLn = x.unaryMinus().plusInPlace(1).logInPlace().timesInPlace(distributionShape2.minus(1));

        final DoubleTensor betaFunction = lnGammaAlpha.plusInPlace(lnGammaBeta).minusInPlace(alphaPlusBetaLnGamma);

        return alphaMinusOneTimesLnX.plusInPlace(betaMinusOneTimesOneMinusXLn).minusInPlace(betaFunction);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor oneMinusX = x.unaryMinus().plusInPlace(1);
        final DoubleTensor digammaAlphaPlusBeta = distributionShape1.plus(distributionShape2).applyInPlace(Gamma::digamma);
        final DoubleTensor alphaMinusOneDivX = x.reciprocal().timesInPlace(distributionShape1.minus(1));

        final DoubleTensor dLogPdx = alphaMinusOneDivX.minusInPlace(oneMinusX.reciprocal().timesInPlace(distributionShape2.minus(1)));
        final DoubleTensor dLogPda = x.log().plusInPlace(digammaAlphaPlusBeta.minus(distributionShape1.apply(Gamma::digamma)));
        final DoubleTensor dLogPdb = oneMinusX.logInPlace().plusInPlace(digammaAlphaPlusBeta.minusInPlace(distributionShape2.apply(Gamma::digamma)));

        return new Diffs()
            .put(A, dLogPda)
            .put(B, dLogPdb)
            .put(X, dLogPdx);
    }
}

package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.Distribution;
import static io.improbable.keanu.distributions.dual.Diffs.A;
import static io.improbable.keanu.distributions.dual.Diffs.B;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import org.apache.commons.math3.special.Gamma;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Beta implements ContinuousDistribution {

    private final DoubleTensor alpha;
    private final DoubleTensor beta;
    private final DoubleTensor xMin;
    private final DoubleTensor xMax;

    public static ContinuousDistribution withParameters(DoubleTensor alpha, DoubleTensor beta, DoubleTensor xMin, DoubleTensor xMax) {
        return new Beta(alpha, beta, xMin, xMax);
    }

    private Beta(DoubleTensor alpha, DoubleTensor beta, DoubleTensor xMin, DoubleTensor xMax) {
        this.alpha = alpha;
        this.beta = beta;
        this.xMin = xMin;
        this.xMax = xMax;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {

        final DoubleTensor y1 = random.nextGamma(shape, DoubleTensor.ONE_SCALAR, alpha);
        final DoubleTensor y2 = random.nextGamma(shape, DoubleTensor.ONE_SCALAR, beta);

        final DoubleTensor range = xMax.minus(xMin);
        final DoubleTensor y1PlusY2 = y1.plus(y2);

        final DoubleTensor lessThan = xMax.minus(y2.div(y1PlusY2).timesInPlace(range));
        final DoubleTensor greaterThan = xMin.plus(y1.div(y1PlusY2).timesInPlace(range));

        final DoubleTensor lessMask = alpha.getLessThanMask(beta);
        final DoubleTensor greaterMask = alpha.getGreaterThanOrEqualToMask(beta);

        return lessMask.timesInPlace(lessThan).plusInPlace(greaterMask.timesInPlace(greaterThan));
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor lnGammaAlpha = alpha.apply(Gamma::logGamma);
        final DoubleTensor lnGammaBeta = beta.apply(Gamma::logGamma);
        final DoubleTensor alphaPlusBetaLnGamma = (alpha.plus(beta)).applyInPlace(Gamma::logGamma);
        final DoubleTensor alphaMinusOneTimesLnX = x.log().timesInPlace(alpha.minus(1));
        final DoubleTensor betaMinusOneTimesOneMinusXLn = x.unaryMinus().plusInPlace(1).logInPlace().timesInPlace(beta.minus(1));

        final DoubleTensor betaFunction = lnGammaAlpha.plusInPlace(lnGammaBeta).minusInPlace(alphaPlusBetaLnGamma);

        return alphaMinusOneTimesLnX.plusInPlace(betaMinusOneTimesOneMinusXLn).minusInPlace(betaFunction);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor oneMinusX = x.unaryMinus().plusInPlace(1);
        final DoubleTensor digammaAlphaPlusBeta = alpha.plus(beta).applyInPlace(Gamma::digamma);
        final DoubleTensor alphaMinusOneDivX = x.reciprocal().timesInPlace(alpha.minus(1));

        final DoubleTensor dLogPdx = alphaMinusOneDivX.minusInPlace(oneMinusX.reciprocal().timesInPlace(beta.minus(1)));
        final DoubleTensor dLogPda = x.log().plusInPlace(digammaAlphaPlusBeta.minus(alpha.apply(Gamma::digamma)));
        final DoubleTensor dLogPdb = oneMinusX.logInPlace().plusInPlace(digammaAlphaPlusBeta.minusInPlace(beta.apply(Gamma::digamma)));

        return new Diffs()
            .put(A, dLogPda)
            .put(B, dLogPdb)
            .put(X, dLogPdx);
    }

    @Override
    public DoubleTensor computeKLDivergence(Distribution<DoubleTensor> q) {
        if (q instanceof Beta) {
            DoubleTensor qAlpha = ((Beta) q).alpha;
            DoubleTensor qBeta = ((Beta) q).beta;

            DoubleTensor pTotal = alpha.plus(beta);
            DoubleTensor qTotal = qAlpha.plus(qBeta);

            DoubleTensor digammaPAlpha = alpha.apply(Gamma::digamma);
            DoubleTensor digammaPBeta = beta.apply(Gamma::digamma);
            DoubleTensor digammaPTotal = pTotal.apply(Gamma::digamma);

            DoubleTensor digammaQAlpha = qAlpha.apply(Gamma::digamma);
            DoubleTensor digammaQBeta = qBeta.apply(Gamma::digamma);
            DoubleTensor digammaQTotal = qTotal.apply(Gamma::digamma);

            DoubleTensor pLogNormalization = digammaPBeta.plus(digammaPAlpha).minusInPlace(digammaPTotal);
            DoubleTensor qLogNormalization = digammaQBeta.plus(digammaQAlpha).minusInPlace(digammaQTotal);

            DoubleTensor deltaLogNormalization = qLogNormalization.minusInPlace(pLogNormalization);
            DoubleTensor deltaAlpha = qAlpha.minus(alpha);
            DoubleTensor deltaBeta = qBeta.minus(beta);
            DoubleTensor deltaTotal = qTotal.minus(pTotal);

            return deltaLogNormalization
                .minusInPlace(digammaPAlpha.timesInPlace(deltaAlpha))
                .minusInPlace(digammaPBeta.timesInPlace(deltaBeta))
                .plusInPlace(digammaPTotal.timesInPlace(deltaTotal));
        } else {
            return ContinuousDistribution.super.computeKLDivergence(q);
        }
    }
}

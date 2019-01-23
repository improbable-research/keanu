package io.improbable.keanu.distributions.continuous;

import com.google.common.base.Preconditions;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LogProbGraph.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.distributions.hyperparam.Diffs.A;
import static io.improbable.keanu.distributions.hyperparam.Diffs.B;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

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
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        Preconditions.checkArgument(alpha.greaterThan(0.).allTrue() && beta.greaterThan(0.).allTrue(),
            "alpha and beta must be positive. alpha: " + alpha + " beta: " + beta);

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
        final DoubleTensor lnGammaAlpha = alpha.logGamma();
        final DoubleTensor lnGammaBeta = beta.logGamma();
        final DoubleTensor alphaPlusBetaLnGamma = (alpha.plus(beta)).logGammaInPlace();
        final DoubleTensor alphaMinusOneTimesLnX = x.log().timesInPlace(alpha.minus(1));
        final DoubleTensor betaMinusOneTimesOneMinusXLn = x.unaryMinus().plusInPlace(1).logInPlace().timesInPlace(beta.minus(1));

        final DoubleTensor betaFunction = lnGammaAlpha.plusInPlace(lnGammaBeta).minusInPlace(alphaPlusBetaLnGamma);

        return alphaMinusOneTimesLnX.plusInPlace(betaMinusOneTimesOneMinusXLn).minusInPlace(betaFunction);
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex alpha, DoublePlaceholderVertex beta) {
        final DoubleVertex lnGammaAlpha = alpha.logGamma();
        final DoubleVertex lnGammaBeta = beta.logGamma();
        final DoubleVertex alphaPlusBetaLnGamma = (alpha.plus(beta)).logGamma();
        final DoubleVertex alphaMinusOneTimesLnX = x.log().times(alpha.minus(1.));
        final DoubleVertex betaMinusOneTimesOneMinusXLn = x.unaryMinus().plus(1.).log().times(beta.minus(1.));

        final DoubleVertex betaFunction = lnGammaAlpha.plus(lnGammaBeta).minus(alphaPlusBetaLnGamma);

        return alphaMinusOneTimesLnX.plus(betaMinusOneTimesOneMinusXLn).minus(betaFunction);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor oneMinusX = x.unaryMinus().plusInPlace(1);
        final DoubleTensor digammaAlphaPlusBeta = alpha.plus(beta).digammaInPlace();
        final DoubleTensor alphaMinusOneDivX = x.reciprocal().timesInPlace(alpha.minus(1));

        final DoubleTensor dLogPdx = alphaMinusOneDivX.minusInPlace(oneMinusX.reciprocal().timesInPlace(beta.minus(1)));
        final DoubleTensor dLogPda = x.log().plusInPlace(digammaAlphaPlusBeta.minus(alpha.digamma()));
        final DoubleTensor dLogPdb = oneMinusX.logInPlace().plusInPlace(digammaAlphaPlusBeta.minusInPlace(beta.digamma()));

        return new Diffs()
            .put(A, dLogPda)
            .put(B, dLogPdb)
            .put(X, dLogPdx);
    }
}

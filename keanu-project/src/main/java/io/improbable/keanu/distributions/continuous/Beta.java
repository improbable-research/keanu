package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.special.Gamma;

public class Beta {

    public static DoubleTensor sample(int[] shape,
                                      DoubleTensor alpha,
                                      DoubleTensor beta,
                                      DoubleTensor xMin,
                                      DoubleTensor xMax,
                                      KeanuRandom random) {

        final DoubleTensor y1 = random.nextGamma(shape, DoubleTensor.ZERO_SCALAR, DoubleTensor.ONE_SCALAR, alpha);
        final DoubleTensor y2 = random.nextGamma(shape, DoubleTensor.ZERO_SCALAR, DoubleTensor.ONE_SCALAR, beta);

        final DoubleTensor range = xMax.minus(xMin);
        final DoubleTensor y1PlusY2 = y1.plus(y2);

        final DoubleTensor lessThan = xMax.minus(y2.div(y1PlusY2).timesInPlace(range));
        final DoubleTensor greaterThan = xMin.plus(y1.div(y1PlusY2).timesInPlace(range));

        final DoubleTensor lessMask = alpha.getLessThanMask(beta);
        final DoubleTensor greaterMask = alpha.getGreaterThanOrEqualToMask(beta);

        return lessMask.timesInPlace(lessThan).plusInPlace(greaterMask.timesInPlace(greaterThan));
    }

    public static DoubleTensor logPdf(DoubleTensor alpha, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor lnGammaAlpha = alpha.apply(Gamma::logGamma);
        final DoubleTensor lnGammaBeta = beta.apply(Gamma::logGamma);
        final DoubleTensor alphaPlusBetaLnGamma = (alpha.plus(beta)).applyInPlace(Gamma::logGamma);
        final DoubleTensor alphaMinusOneTimesLnX = x.log().timesInPlace(alpha.minus(1));
        final DoubleTensor betaMinusOneTimesOneMinusXLn = x.unaryMinus().plusInPlace(1).logInPlace().timesInPlace(beta.minus(1));

        final DoubleTensor betaFunction = lnGammaAlpha.plusInPlace(lnGammaBeta).minusInPlace(alphaPlusBetaLnGamma);

        return alphaMinusOneTimesLnX.plusInPlace(betaMinusOneTimesOneMinusXLn).minusInPlace(betaFunction);
    }

    public static DiffLogP dlnPdf(DoubleTensor alpha, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor oneMinusX = x.unaryMinus().plusInPlace(1);
        final DoubleTensor digammaAlphaPlusBeta = alpha.plus(beta).applyInPlace(Gamma::digamma);
        final DoubleTensor alphaMinusOneDivX = x.reciprocal().timesInPlace(alpha.minus(1));

        final DoubleTensor dLogPdx = alphaMinusOneDivX.minusInPlace(oneMinusX.reciprocal().timesInPlace(beta.minus(1)));
        final DoubleTensor dLogPda = x.log().plusInPlace(digammaAlphaPlusBeta.minus(alpha.apply(Gamma::digamma)));
        final DoubleTensor dLogPdb = oneMinusX.logInPlace().plusInPlace(digammaAlphaPlusBeta.minusInPlace(beta.apply(Gamma::digamma)));

        return new DiffLogP(dLogPda, dLogPdb, dLogPdx);
    }

    public static class DiffLogP {
        public final DoubleTensor dLogPdalpha;
        public final DoubleTensor dLogPdbeta;
        public final DoubleTensor dLogPdx;

        public DiffLogP(DoubleTensor dLogPdalpha, DoubleTensor dLogPdbeta, DoubleTensor dLogPdx) {
            this.dLogPdalpha = dLogPdalpha;
            this.dLogPdbeta = dLogPdbeta;
            this.dLogPdx = dLogPdx;
        }
    }

}

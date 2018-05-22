package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.Nd4jDoubleTensor;
import org.apache.commons.math3.special.Gamma;

public class TensorBeta {

    public static DoubleTensor sample(int[] shape,
                                    DoubleTensor alpha,
                                    DoubleTensor beta,
                                    DoubleTensor xMin,
                                    DoubleTensor xMax,
                                    KeanuRandom random) {

        final Nd4jDoubleTensor zeros = Nd4jDoubleTensor.zeros(shape);
        final Nd4jDoubleTensor ones = Nd4jDoubleTensor.ones(shape);

        final DoubleTensor y1 = random.nextGamma(shape,
            zeros,
            ones,
            alpha
        );

        final DoubleTensor y2 = random.nextGamma(shape,
            zeros,
            ones,
            beta
        );

        final DoubleTensor xMaxMinusxMin = xMax.minus(xMin);
        final DoubleTensor y1PlusY2 = y1.plus(y2);

        final DoubleTensor lessThan = xMax.minus(y2.div(y1PlusY2).timesInPlace(xMaxMinusxMin));
        final DoubleTensor greaterThan = xMin.plus(y1.div(y1PlusY2).timesInPlace(xMaxMinusxMin));

        final DoubleTensor lessMask = alpha.getLessThanMask(beta);
        final DoubleTensor greaterMask = alpha.getGreaterThanOrEqualToMask(beta);

        return lessMask.times(lessThan).plus(greaterMask.times(greaterThan));
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

    public static Diff dlnPdf(DoubleTensor alpha, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor oneMinusX = x.unaryMinus().plusInPlace(1);
        final DoubleTensor digammaAlphaPlusBeta = alpha.plus(beta).applyInPlace(Gamma::digamma);
        final DoubleTensor alphaMinusOneDivX = x.reciprocal().timesInPlace(alpha.minus(1));

        final DoubleTensor dPdx = alphaMinusOneDivX.minusInPlace(oneMinusX.reciprocal().timesInPlace(beta.minus(1)));
        final DoubleTensor dPda = x.log().plusInPlace(digammaAlphaPlusBeta.minus(alpha.apply(Gamma::digamma)));
        final DoubleTensor dPdb = oneMinusX.logInPlace().plusInPlace(digammaAlphaPlusBeta.minusInPlace(beta.apply(Gamma::digamma)));

        return new Diff(dPda, dPdb, dPdx);
    }

    public static class Diff {
        public final DoubleTensor dPdalpha;
        public final DoubleTensor dPdbeta;
        public final DoubleTensor dPdx;

        public Diff(DoubleTensor dPdalpha, DoubleTensor dPdbeta, DoubleTensor dPdx) {
            this.dPdalpha = dPdalpha;
            this.dPdbeta = dPdbeta;
            this.dPdx = dPdx;
        }
    }

}

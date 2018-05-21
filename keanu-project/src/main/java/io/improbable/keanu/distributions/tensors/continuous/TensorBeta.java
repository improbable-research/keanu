package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.Nd4jDoubleTensor;
import org.apache.commons.math3.special.Gamma;

public class TensorBeta {

    public static DoubleTensor sample(int[] shape, DoubleTensor alpha, DoubleTensor beta, DoubleTensor xMin, DoubleTensor xMax, KeanuRandom random) {
        final DoubleTensor y1 = random.nextGamma(shape, Nd4jDoubleTensor.zeros(shape), Nd4jDoubleTensor.ones(shape), alpha);
        final DoubleTensor y2 = random.nextGamma(shape, Nd4jDoubleTensor.zeros(shape), Nd4jDoubleTensor.ones(shape), beta);

        final DoubleTensor xMaxMinusxMin = xMax.minus(xMin);
        final DoubleTensor y1PlusY2 = y1.plus(y2);

        final DoubleTensor lessThan = xMax.minus(y2.div(y1PlusY2).times(xMaxMinusxMin));
        final DoubleTensor greaterThan = xMin.plus(y1.div(y1PlusY2).times(xMaxMinusxMin));

        final DoubleTensor lessMask = alpha.getLessThanMask(beta);
        final DoubleTensor greaterMask = alpha.getGreaterThanOrEqualToMask(beta);

        return lessMask.times(lessThan).plus(greaterMask.times(greaterThan));
    }

    public static DoubleTensor logPdf(DoubleTensor alpha, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor logGammaAlpha = alpha.apply(Gamma::logGamma);
        final DoubleTensor logGammaBeta = beta.apply(Gamma::logGamma);
        final DoubleTensor alphaPlusBetaLogGamma = (alpha.plus(beta)).apply(Gamma::logGamma);
        final DoubleTensor alphaMinusOneTimesLnX = alpha.minus(1).times(x.log());
        final DoubleTensor betaMinusOneTimesOneMinusXLog = beta.minus(1).times(x.unaryMinus().plus(1).log());

        final DoubleTensor betaFunction = logGammaAlpha.plus(logGammaBeta).minus(alphaPlusBetaLogGamma);

        return alphaMinusOneTimesLnX.plus(betaMinusOneTimesOneMinusXLog).minus(betaFunction);
    }

    public static Diff dlnPdf(DoubleTensor alpha, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor oneMinusX = x.unaryMinus().plus(1);
        final DoubleTensor digammaAlphaPlusBeta = alpha.plus(beta).apply(Gamma::digamma);
        final DoubleTensor alphaMinusOneDivX = x.reciprocal().times(alpha.minus(1));

        final DoubleTensor dPdx = alphaMinusOneDivX.minus(oneMinusX.reciprocal().times(beta.minus(1)));
        final DoubleTensor dPda = x.log().plus(digammaAlphaPlusBeta.minus(alpha.apply(Gamma::digamma)));
        final DoubleTensor dPdb = oneMinusX.log().plus(digammaAlphaPlusBeta.minus(beta.apply(Gamma::digamma)));

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

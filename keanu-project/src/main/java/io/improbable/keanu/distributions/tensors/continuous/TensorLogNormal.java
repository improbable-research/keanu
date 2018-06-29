package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.distributions.continuous.Gaussian.LN_SQRT_2PI;

public class TensorLogNormal {

    private TensorLogNormal() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor mu, DoubleTensor sigma, KeanuRandom random) {
        return TensorGaussian.sample(shape, mu, sigma, random).expInPlace();
    }

    public static DoubleTensor logPdf(DoubleTensor mu, DoubleTensor sigma, DoubleTensor x) {
        final DoubleTensor lnSigmaX = sigma.times(x).log();
        final DoubleTensor lnXMinusMuSquared = x.log().minusInPlace(mu).powInPlace(2);
        final DoubleTensor lnXMinusMuSquaredOver2Variance = lnXMinusMuSquared.divInPlace(sigma.pow(2).timesInPlace(2.0));
        return lnXMinusMuSquaredOver2Variance.plusInPlace(lnSigmaX).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    public static TensorLogNormal.Diff dlnPdf(DoubleTensor mu, DoubleTensor sigma, DoubleTensor x) {
        final DoubleTensor variance = sigma.pow(2);
        final DoubleTensor lnXMinusMu = x.log().minusInPlace(mu);

        final DoubleTensor dlnP_dmu = lnXMinusMu.div(variance);
        final DoubleTensor dlnP_dx = dlnP_dmu.plus(1.0).unaryMinus().divInPlace(x);
        final DoubleTensor dlnP_dsigma = lnXMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(sigma))
            .minusInPlace(sigma.reciprocal());

        return new TensorLogNormal.Diff(dlnP_dmu, dlnP_dsigma, dlnP_dx);
    }

    public static class Diff {
        public final DoubleTensor dPdmu;
        public final DoubleTensor dPdsigma;
        public final DoubleTensor dPdx;

        public Diff(DoubleTensor dPdmu, DoubleTensor dPdsigma, DoubleTensor dPdx) {
            this.dPdmu = dPdmu;
            this.dPdsigma = dPdsigma;
            this.dPdx = dPdx;
        }
    }
}

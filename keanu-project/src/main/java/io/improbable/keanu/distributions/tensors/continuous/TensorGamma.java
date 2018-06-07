package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.special.Gamma;

public class TensorGamma {

    private TensorGamma() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor a, DoubleTensor theta, DoubleTensor k, KeanuRandom random) {
        return random.nextGamma(shape, a, theta, k);
    }

    public static DoubleTensor logPdf(DoubleTensor a, DoubleTensor theta, DoubleTensor k, DoubleTensor x) {
        final DoubleTensor aMinusXOverTheta = a.minus(x).divInPlace(theta);
        final DoubleTensor kLnTheta = k.times(theta.log());
        final DoubleTensor xMinusAPowKMinus1 = x.minus(a).powInPlace(k.minus(1));
        final DoubleTensor lnXMinusAToKMinus1 = ((xMinusAPowKMinus1).divInPlace(k.apply(Gamma::gamma))).logInPlace();
        return aMinusXOverTheta.minusInPlace(kLnTheta).plusInPlace(lnXMinusAToKMinus1);
    }

    public static Diff dlnPdf(DoubleTensor a, DoubleTensor theta, DoubleTensor k, DoubleTensor x) {
        final DoubleTensor xMinusA = x.minus(a);
        final DoubleTensor aMinusX = a.minus(x);
        final DoubleTensor kMinus1 = k.minus(1.);
        final DoubleTensor oneOverTheta = theta.reciprocal();

        final DoubleTensor dPdx = kMinus1.div(xMinusA).minusInPlace(oneOverTheta);
        final DoubleTensor dPda = kMinus1.div(aMinusX).plusInPlace(oneOverTheta);
        final DoubleTensor dPdtheta = theta.times(k).plus(aMinusX).divInPlace(theta.pow(2.)).unaryMinusInPlace();
        final DoubleTensor dPdk = xMinusA.logInPlace().minusInPlace(theta.log()).minusInPlace(k.apply(Gamma::digamma));

        return new Diff(dPda, dPdtheta, dPdk, dPdx);
    }

    public static class Diff {
        public final DoubleTensor dPda;
        public final DoubleTensor dPdtheta;
        public final DoubleTensor dPdk;
        public final DoubleTensor dPdx;

        public Diff(DoubleTensor dPda, DoubleTensor dPdtheta, DoubleTensor dPdk, DoubleTensor dPdx) {
            this.dPda = dPda;
            this.dPdtheta = dPdtheta;
            this.dPdk = dPdk;
            this.dPdx = dPdx;
        }
    }

}

package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.apache.commons.math3.special.Gamma;

import java.util.Random;

public class TensorGamma {

    private TensorGamma() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor a, DoubleTensor theta, DoubleTensor k, KeanuRandom random) {
        return random.nextGamma(shape, a, theta, k, new Random());
    }

    public static DoubleTensor logPdf(DoubleTensor a, DoubleTensor theta, DoubleTensor k, DoubleTensor x) {
        final DoubleTensor aMinusXOverTheta = a.minus(x).div(theta);
        final DoubleTensor kLnTheta = k.times(theta.log());
        final DoubleTensor lnXMinusAToKMinus1 = ((x.minus(a).pow(k.minus(1.))).div(k.apply(Gamma::gamma))).log();
        return aMinusXOverTheta.minus(kLnTheta).plus(lnXMinusAToKMinus1);
    }

    public static Diff dlnPdf(DoubleTensor a, DoubleTensor theta, DoubleTensor k, DoubleTensor x) {
        final DoubleTensor xMinusA = x.minus(a);
        final DoubleTensor kMinus1 = k.minus(1.);
        final DoubleTensor oneOverTheta = theta.reciprocal();

        final DoubleTensor dPdx = kMinus1.div(xMinusA).minus(oneOverTheta);
        final DoubleTensor dPda = kMinus1.div(a.minus(x)).plus(oneOverTheta);
        final DoubleTensor dPdtheta = a.plus(theta.times(k)).minus(x).div(theta.pow(2.)).unaryMinus();
        final DoubleTensor dPdk = xMinusA.log().minus(theta.log()).minus(k.apply(Gamma::digamma));

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

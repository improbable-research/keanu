package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Logistic {

    /**
     * @param shape  shape of tensor returned
     * @param mu     location parameter (any real number)
     * @param s      scale parameter (b greater than 0)
     * @param random source or randomness
     * @return a sample from the distribution
     */
    public static DoubleTensor sample(int[] shape, DoubleTensor mu, DoubleTensor s, KeanuRandom random) {
        return random.nextDouble(shape).reciprocalInPlace().minusInPlace(1).logInPlace().timesInPlace(mu.minus(s));
    }

    /**
     * @param mu location parameter (any real number)
     * @param s  scale parameter (b greater than 0)
     * @param x  at value
     * @return the density at x
     */
    public static DoubleTensor logPdf(DoubleTensor mu, DoubleTensor s, DoubleTensor x) {
        final DoubleTensor xMinusAOverB = x.minus(mu).divInPlace(s);
        final DoubleTensor ln1OverB = s.reciprocal().logInPlace();

        return xMinusAOverB.plus(ln1OverB).minusInPlace(
            xMinusAOverB.expInPlace().plusInPlace(1).logInPlace().timesInPlace(2)
        );
    }

    public static Diff dlnPdf(DoubleTensor mu, DoubleTensor s, DoubleTensor x) {
        final DoubleTensor expAOverB = mu.div(s).expInPlace();
        final DoubleTensor expXOverB = x.div(s).expInPlace();
        final DoubleTensor expPlus = expAOverB.plus(expXOverB);
        final DoubleTensor bTimesExpAOverB = expAOverB.times(s);
        final DoubleTensor bTimesExpXOverB = expXOverB.times(s);

        final DoubleTensor dPda = expXOverB.minus(expAOverB).divInPlace(s.times(expPlus));
        final DoubleTensor dPdx = expAOverB.minus(expXOverB).divInPlace(bTimesExpAOverB.plus(bTimesExpXOverB));

        final DoubleTensor numeratorPartOne = mu.times(expXOverB).plusInPlace(x.times(expAOverB)).plusInPlace(
            mu.times(expAOverB.unaryMinus())
        );
        final DoubleTensor numeratorPartTwo = bTimesExpAOverB.plus(bTimesExpXOverB).minusInPlace(x.times(expXOverB));
        final DoubleTensor denominator = s.pow(2).timesInPlace(expPlus);

        final DoubleTensor dPdb = numeratorPartOne.plus(numeratorPartTwo).divInPlace(denominator).unaryMinusInPlace();

        return new Diff(dPda, dPdb, dPdx);
    }

    public static class Diff {
        public final DoubleTensor dPdmu;
        public final DoubleTensor dPds;
        public final DoubleTensor dPdx;

        public Diff(DoubleTensor dPdmu, DoubleTensor dPds, DoubleTensor dPdx) {
            this.dPdmu = dPdmu;
            this.dPds = dPds;
            this.dPdx = dPdx;
        }
    }

}

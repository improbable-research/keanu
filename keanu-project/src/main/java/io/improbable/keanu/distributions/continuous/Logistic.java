package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.tensor.Tensor.SCALAR_SHAPE;
import static io.improbable.keanu.tensor.TensorShape.concat;

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

    public static DiffLogP dlnPdf(DoubleTensor mu, DoubleTensor s, DoubleTensor x) {
        final DoubleTensor expAOverB = mu.div(s).expInPlace();
        final DoubleTensor expXOverB = x.div(s).expInPlace();
        final DoubleTensor expPlus = expAOverB.plus(expXOverB);
        final DoubleTensor bTimesExpAOverB = expAOverB.times(s);
        final DoubleTensor bTimesExpXOverB = expXOverB.times(s);

        DoubleTensor dLogPdmu = expXOverB.minus(expAOverB).divInPlace(s.times(expPlus));
        DoubleTensor dLogPdx = expAOverB.minus(expXOverB).divInPlace(bTimesExpAOverB.plus(bTimesExpXOverB));

        final DoubleTensor numeratorPartOne = mu.times(expXOverB).plusInPlace(x.times(expAOverB)).plusInPlace(
            mu.times(expAOverB.unaryMinus())
        );
        final DoubleTensor numeratorPartTwo = bTimesExpAOverB.plus(bTimesExpXOverB).minusInPlace(x.times(expXOverB));
        final DoubleTensor denominator = s.pow(2).timesInPlace(expPlus);

        DoubleTensor dLogPds = numeratorPartOne.plus(numeratorPartTwo).divInPlace(denominator).unaryMinusInPlace();

        dLogPdmu = dLogPdmu.reshape(concat(SCALAR_SHAPE, dLogPdmu.getShape()));
        dLogPds = dLogPds.reshape(concat(SCALAR_SHAPE, dLogPds.getShape()));
        dLogPdx = dLogPdx.reshape(concat(SCALAR_SHAPE, dLogPdx.getShape()));

        return new DiffLogP(dLogPdmu, dLogPds, dLogPdx);
    }

    public static class DiffLogP {
        public final DoubleTensor dLogPdmu;
        public final DoubleTensor dLogPds;
        public final DoubleTensor dLogPdx;

        public DiffLogP(DoubleTensor dLogPdmu, DoubleTensor dLogPds, DoubleTensor dLogPdx) {
            this.dLogPdmu = dLogPdmu;
            this.dLogPds = dLogPds;
            this.dLogPdx = dLogPdx;
        }
    }

}

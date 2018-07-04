package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.special.Gamma;

import static java.lang.Math.PI;
import static java.lang.Math.log;

/**
 * Student T Distribution
 * https://en.wikipedia.org/wiki/Student%27s_t-distribution#Sampling_distribution
 */
public class StudentT {

    private static final double HALF_LOG_PI = log(PI) / 2;

    /**
     * Computer Generation of Statistical Distributions
     * by Richard Saucier
     * ARL-TR-2168 March 2000
     * 5.1.23 page 36
     *
     * @param shape  tensor shape of returned samples
     * @param v      Degrees of Freedom
     * @param random random number generator (RNG) seed
     * @return sample of Student T distribution
     */
    public static DoubleTensor sample(int[] shape, IntegerTensor v, KeanuRandom random) {

        DoubleTensor chi2Samples = ChiSquared.sample(shape, v, random);
        return random.nextGaussian(shape).divInPlace(chi2Samples.divInPlace(v.toDouble()).sqrtInPlace());
    }

    /**
     * @param v Degrees of Freedom
     * @param t random variable
     * @return Log of the Probability Density Function
     */
    public static DoubleTensor logPdf(IntegerTensor v, DoubleTensor t) {

        DoubleTensor vAsDouble = v.toDouble();
        DoubleTensor halfVPlusOne = vAsDouble.plus(1).divInPlace(2);

        DoubleTensor logGammaHalfVPlusOne = halfVPlusOne.apply(Gamma::logGamma);
        DoubleTensor logGammaHalfV = vAsDouble.div(2).applyInPlace(Gamma::logGamma);
        DoubleTensor halfLogV = vAsDouble.log().divInPlace(2);

        return logGammaHalfVPlusOne
            .minusInPlace(halfLogV)
            .minusInPlace(HALF_LOG_PI)
            .minusInPlace(logGammaHalfV)
            .minusInPlace(
                halfVPlusOne.timesInPlace(
                    t.pow(2).divInPlace(vAsDouble).plusInPlace(1).logInPlace()
                )
            );
    }

    /**
     * @param v Degrees of Freedom
     * @param t random variable
     * @return Differential of the Log of the Probability Density Function
     */
    public static Diff dLogPdf(IntegerTensor v, DoubleTensor t) {

        DoubleTensor vAsDouble = v.toDouble();
        DoubleTensor dPdt = t.unaryMinus()
            .timesInPlace(vAsDouble.plus(1.0))
            .divInPlace(
                t.pow(2).plusInPlace(vAsDouble)
            );

        return new Diff(dPdt);
    }

    /**
     * Differential Equation Class to store result of d/dv and d/dt
     */
    public static class Diff {
        public DoubleTensor dPdt;

        public Diff(DoubleTensor dPdt) {
            this.dPdt = dPdt;
        }
    }
}
